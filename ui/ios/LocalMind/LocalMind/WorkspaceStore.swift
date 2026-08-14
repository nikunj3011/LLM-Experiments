import Foundation
import Observation

/// Every top-level screen understood by the app shell and persistent tab bar.
enum AppDestination: String, CaseIterable, Hashable, Identifiable { case chat, image, video, audio, gallery, settings; var id: String { rawValue }; var title: String { switch self { case .chat: "Chat"; case .image: "Image studio"; case .video: "Motion studio"; case .audio: "Sound lab"; case .gallery: "Gallery"; case .settings: "Settings" } } }
/// Backend-supported generation quality/speed selection sent alongside each message.
enum GenerationMode: String, CaseIterable, Identifiable, Codable { case flash, thinking; var id: String { rawValue }; var title: String { self == .flash ? "Flash" : "Think" } }
enum MessageRole: String, Codable { case user, assistant, system }
enum MessageStatus: String, Codable { case sending, streaming, complete, failed }
struct ChatMessage: Identifiable, Codable, Equatable { var id = UUID(); let role: MessageRole; var content: String; var fileName: String?; var modelUsed: String?; var modeUsed: GenerationMode?; var status: MessageStatus = .complete }
struct ChatAttachment: Equatable { let url: URL; let name: String }
struct AIModel: Identifiable, Codable, Hashable { let id: String; let name: String }
struct ChatSession: Identifiable, Hashable { let id: String; let title: String }
struct GalleryAsset: Identifiable, Decodable { let filename: String; let url: URL; var id: String { url.absoluteString } }

/// Main-actor view model: it translates user actions into API calls and publishes UI state.
@MainActor @Observable final class WorkspaceStore {
    /// The app shell reads this to decide which feature is visible.
    var destination: AppDestination = .chat; var messages: [ChatMessage] = []; var sessions: [ChatSession] = []; var models: [AIModel] = []; var selectedModelID = ""; var selectedSession: ChatSession?; var generationMode: GenerationMode = .flash; var gallery: [GalleryAsset] = []; var isGenerating = false; var isShowingError = false; var errorMessage = ""
    /// Persist a user-selected local endpoint so physical devices can use a Mac's LAN address.
    var apiBaseURL: String { didSet { UserDefaults.standard.set(apiBaseURL, forKey: "apiBaseURL") } }
    private let api = LocalMindAPI(); private var generationTask: Task<Void, Never>?
    init() { apiBaseURL = UserDefaults.standard.string(forKey: "apiBaseURL") ?? "http://127.0.0.1:8000/api" }
    var activeModel: AIModel? { models.first { $0.id == selectedModelID } }
    /// Load independent sidebar/model data concurrently to keep launch responsive.
    func refresh() async {
        async let fetchedSessions = api.sessions(baseURL: apiBaseURL)
        async let fetchedModels = api.models(baseURL: apiBaseURL)

        do {
            let (loadedSessions, loadedModels) = try await (fetchedSessions, fetchedModels)
            self.sessions = loadedSessions
            self.models = loadedModels

            if self.selectedModelID.isEmpty {
                self.selectedModelID = loadedModels.first?.id ?? ""
            }
        } catch {
            report(error)
        }
    }
    func newChat() { selectedSession = nil; messages = []; destination = .chat }
    func select(_ session: ChatSession) async { do { messages = try await api.loadSession(session.id, baseURL: apiBaseURL); selectedSession = session; destination = .chat } catch { report(error) } }
    func deleteSessions(at offsets: IndexSet) async { for index in offsets { do { try await api.deleteSession(sessions[index].id, baseURL: apiBaseURL) } catch { report(error) } }; await refresh() }
    func clearVRAM() async { do { try await api.clearVRAM(baseURL: apiBaseURL) } catch { report(error) } }
    func refreshGallery() async { do { gallery = try await api.gallery(baseURL: apiBaseURL) } catch { report(error) } }
    func send(prompt: String, attachment: ChatAttachment?) async { guard !isGenerating else { return }; let user = ChatMessage(role: .user, content: prompt, fileName: attachment?.name, modelUsed: selectedModelID, modeUsed: generationMode, status: .sending); let assistant = ChatMessage(role: .assistant, content: "", modelUsed: selectedModelID, modeUsed: generationMode, status: .streaming); messages += [user, assistant]; isGenerating = true; let history = messages.dropLast().map { $0 }; generationTask = Task { [weak self] in guard let self else { return }; do { try await self.api.stream(prompt: prompt, model: self.selectedModelID, mode: self.generationMode, history: history, sessionID: self.selectedSession?.id, attachment: attachment, baseURL: self.apiBaseURL) { [weak self] token, sessionID in Task { @MainActor in guard let self else { return }; if let sessionID { self.selectedSession = ChatSession(id: sessionID, title: Self.title(for: sessionID)) }; guard let index = self.messages.indices.last else { return }; self.messages[index].content += token } } } catch is CancellationError { } catch { self.messages[self.messages.count - 1].content = "Sorry, I couldn't generate a response. Check that the local runtime is running."; self.messages[self.messages.count - 1].status = .failed; self.report(error) }; self.isGenerating = false; if self.messages.last?.status == .streaming { self.messages[self.messages.count - 1].status = .complete }; await self.refresh() } }
    /// Cancellation propagates into the URLSession streaming task, not just the visible UI.
    func stopGeneration() { generationTask?.cancel(); generationTask = nil; isGenerating = false }
    private func report(_ error: Error) { errorMessage = error.localizedDescription; isShowingError = true }
    static func title(for id: String) -> String { id.replacingOccurrences(of: "chat_", with: "").replacingOccurrences(of: "_", with: " ") }
}
