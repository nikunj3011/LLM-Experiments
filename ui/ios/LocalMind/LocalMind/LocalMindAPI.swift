import Foundation

enum APIError: LocalizedError {
    case invalidURL, invalidResponse, status(Int), decoding
    var errorDescription: String? {
        switch self { case .invalidURL: "The local runtime URL is invalid."; case .invalidResponse: "The local runtime returned an invalid response."; case .status(let code): "The local runtime returned HTTP \(code)."; case .decoding: "The local runtime returned data in an unexpected format." }
    }
}

/// Actor-isolated transport layer. Views and the store never construct URLRequests directly.
actor LocalMindAPI {
    private let session: URLSession = {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 60
        config.timeoutIntervalForResource = 600
        return URLSession(configuration: config)
    }()

    func models(baseURL: String) async throws -> [AIModel] {
        struct Response: Decodable { let models: [AIModel] }
        let response: Response = try await request("/models", baseURL: baseURL)
        return response.models
    }

    func sessions(baseURL: String) async throws -> [ChatSession] {
        struct Response: Decodable { let sessions: [String] }
        let response: Response = try await request("/sessions", baseURL: baseURL)
        return response.sessions.map { ChatSession(id: $0, title: $0.replacingOccurrences(of: "chat_", with: "").replacingOccurrences(of: "_", with: " ")) }
    }

    func loadSession(_ id: String, baseURL: String) async throws -> [ChatMessage] {
        struct Response: Decodable { let history: [ServerMessage] }
        struct Payload: Encodable { let session_id: String }
        let body = try JSONEncoder().encode(Payload(session_id: id))
        let response: Response = try await request("/load_session", method: "POST", body: body, baseURL: baseURL)
        return response.history.map(ChatMessage.init)
    }

    func deleteSession(_ id: String, baseURL: String) async throws {
        let _: EmptyResponse = try await request("/sessions/\(id.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? id)", method: "DELETE", baseURL: baseURL)
    }

    func clearVRAM(baseURL: String) async throws {
        let _: EmptyResponse = try await request("/clear_vram", method: "POST", baseURL: baseURL)
    }

    func gallery(baseURL: String) async throws -> [GalleryAsset] {
        struct Response: Decodable { let images: [GalleryAsset]?; let videos: [GalleryAsset]? }
        let response: Response = try await request("/gallery", baseURL: baseURL)
        return (response.images ?? []) + (response.videos ?? [])
    }

    /// Sends multipart chat input and yields server-sent-event tokens as soon as they arrive.
    /// The caller's Task can cancel this `bytes` loop to stop generation promptly.
    func stream(prompt: String, model: String, mode: GenerationMode, history: [ChatMessage], sessionID: String?, attachment: ChatAttachment?, baseURL: String, onToken: @escaping @Sendable (String, String?) -> Void) async throws {
        let boundary = "LocalMind-\(UUID().uuidString)"
        var request = try makeRequest("/stream", method: "POST", baseURL: baseURL)
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        request.httpBody = try multipart(boundary: boundary, prompt: prompt, model: model, mode: mode, history: history, sessionID: sessionID, attachment: attachment)
        let (bytes, response) = try await session.bytes(for: request)
        guard let http = response as? HTTPURLResponse else { throw APIError.invalidResponse }
        guard (200..<300).contains(http.statusCode) else { throw APIError.status(http.statusCode) }
        for try await line in bytes.lines {
            try Task.checkCancellation()
            guard line.hasPrefix("data: ") else { continue }
            guard let data = String(line.dropFirst(6)).data(using: .utf8), let event = try? JSONDecoder().decode(StreamEvent.self, from: data) else { continue }
            if let token = event.token { onToken(token, event.session_id) }
        }
    }

    /// Shared JSON request path: it centralizes HTTP status and decode failures.
    private func request<T: Decodable>(_ path: String, method: String = "GET", body: Data? = nil, baseURL: String) async throws -> T {
        var request = try makeRequest(path, method: method, baseURL: baseURL)
        if let body { request.setValue("application/json", forHTTPHeaderField: "Content-Type"); request.httpBody = body }
        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw APIError.invalidResponse }
        guard (200..<300).contains(http.statusCode) else { throw APIError.status(http.statusCode) }
        if T.self == EmptyResponse.self { return EmptyResponse() as! T }
        guard let value = try? JSONDecoder().decode(T.self, from: data) else { throw APIError.decoding }
        return value
    }

    private func makeRequest(_ path: String, method: String, baseURL: String) throws -> URLRequest {
        guard let url = URL(string: baseURL.trimmingCharacters(in: CharacterSet(charactersIn: "/")) + path) else { throw APIError.invalidURL }
        var request = URLRequest(url: url); request.httpMethod = method; request.setValue("application/json", forHTTPHeaderField: "Accept")
        return request
    }

    /// Builds the backend's expected form payload and safely opens a Files-picked URL only while reading it.
    private func multipart(boundary: String, prompt: String, model: String, mode: GenerationMode, history: [ChatMessage], sessionID: String?, attachment: ChatAttachment?) throws -> Data {
        var data = Data(); let prefix = "--\(boundary)\r\n"
        func field(_ name: String, _ value: String) { data.append(prefix.data(using: .utf8)!); data.append("Content-Disposition: form-data; name=\"\(name)\"\r\n\r\n\(value)\r\n".data(using: .utf8)!) }
        field("prompt", prompt); field("model", model); field("mode", mode.rawValue)
        field("messages", String(data: try JSONEncoder().encode(history), encoding: .utf8) ?? "[]")
        if let sessionID { field("session_id", sessionID) }
        if let attachment {
            let accessed = attachment.url.startAccessingSecurityScopedResource(); defer { if accessed { attachment.url.stopAccessingSecurityScopedResource() } }
            data.append(prefix.data(using: .utf8)!); data.append("Content-Disposition: form-data; name=\"file\"; filename=\"\(attachment.name)\"\r\nContent-Type: application/octet-stream\r\n\r\n".data(using: .utf8)!); data.append(try Data(contentsOf: attachment.url)); data.append("\r\n".data(using: .utf8)!)
        }
        data.append("--\(boundary)--\r\n".data(using: .utf8)!)
        return data
    }
}

private struct EmptyResponse: Decodable {}
private struct StreamEvent: Decodable { let token: String?; let session_id: String? }
private struct ServerMessage: Decodable { let role: MessageRole; let content: JSONValue; let fileName: String?; let modelUsed: String?; let modeUsed: GenerationMode? }
private extension ChatMessage { init(_ server: ServerMessage) { let cachedPath = ChatAttachment.cachedPath(for: server.fileName); self.init(role: server.role, content: server.content.stringValue, fileName: server.fileName, attachmentPath: cachedPath, attachmentKind: cachedPath.map { AttachmentKind.infer(from: URL(fileURLWithPath: $0)) }, modelUsed: server.modelUsed, modeUsed: server.modeUsed) } }
private enum JSONValue: Decodable { case string(String), other; init(from decoder: Decoder) throws { if let value = try? decoder.singleValueContainer().decode(String.self) { self = .string(value) } else { self = .other } }; var stringValue: String { if case .string(let value) = self { value } else { "" } } }
