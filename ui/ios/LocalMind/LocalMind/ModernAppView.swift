import SwiftUI
import UniformTypeIdentifiers
import AVKit
import UIKit
import PhotosUI
import CoreTransferable

/// The active application shell. It deliberately owns navigation and presentation state in one
/// small place, while `WorkspaceStore` continues to own the backend and conversation state.
struct ModernAppView: View {
    @State private var store = WorkspaceStore()
    @State private var page: ModernPage = .chat
    @State private var isDrawerOpen = false
    @AppStorage("appearance") private var appearance = "system"

    var body: some View {
        ZStack(alignment: .leading) {
            NavigationStack {
                pageContent
                    // This is attached to the app shell—not to a single screen—so the
                    // navigation stays available while browsing the library or settings.
                    .safeAreaInset(edge: .bottom, spacing: 8) {
                        ModernBottomNavigation(page: $page)
                    }
                    .toolbar {
                        ToolbarItem(placement: .topBarLeading) {
                            Button { withAnimation(.spring(response: 0.36, dampingFraction: 0.84)) { isDrawerOpen = true } } label: {
                                // Keep the navigation controls visually weightless: the content owns the glass,
                                // rather than placing opaque material circles over the page background.
                                Image(systemName: "line.3.horizontal").font(.headline).frame(width: 42, height: 42)
                            }
                        }
                        ToolbarItem(placement: .topBarTrailing) {
                            Menu { Picker("Model", selection: $store.selectedModelID) { ForEach(store.models) { Text($0.name).tag($0.id) } } } label: {
                                Image(systemName: "cpu").font(.headline).frame(width: 42, height: 42)
                            }
                        }
                    }
            }
            .simultaneousGesture(DragGesture(minimumDistance: 16).onEnded { value in
                guard value.startLocation.x < 28, value.translation.width > 80 else { return }
                withAnimation(.spring(response: 0.36, dampingFraction: 0.84)) { isDrawerOpen = true }
            })

            if isDrawerOpen {
                Color.black.opacity(0.28).ignoresSafeArea().onTapGesture { closeDrawer() }.transition(.opacity)
                ModernDrawer(store: store, page: $page, close: closeDrawer)
                    .frame(width: 330).frame(maxHeight: .infinity)
                    .background(.regularMaterial)
                    .clipShape(UnevenRoundedRectangle(bottomTrailingRadius: 32, topTrailingRadius: 32))
                    .shadow(color: .black.opacity(0.22), radius: 20, x: 8)
                    .transition(.move(edge: .leading))
                    .zIndex(2)
            }
        }
        .preferredColorScheme(colorScheme)
        .tint(.primary)
        .task { await store.refresh() }
        .alert("LocalMind", isPresented: $store.isShowingError) { Button("OK", role: .cancel) {} } message: { Text(store.errorMessage) }
    }

    @ViewBuilder private var pageContent: some View {
        switch page {
        case .chat:
            ModernChat(store: store, page: $page)
                .navigationTitle(store.selectedSession?.title ?? "New conversation")
        case .library:
            ModernLibrary(store: store).navigationTitle("Library")
        case .settings:
            ModernSettings(appearance: $appearance, store: store).navigationTitle("Settings")
        }
    }

    private var colorScheme: ColorScheme? { appearance == "light" ? .light : appearance == "dark" ? .dark : nil }
    private func closeDrawer() { withAnimation(.spring(response: 0.32, dampingFraction: 0.86)) { isDrawerOpen = false } }
}

private enum ModernPage: Equatable { case chat, library, settings }

/// A stable full-height recents drawer inspired by modern AI chat apps.
private struct ModernDrawer: View {
    @Bindable var store: WorkspaceStore
    @Binding var page: ModernPage
    let close: () -> Void
    @State private var selectedWorkspace = "Remote"
    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 12) {
                Image(systemName: "sparkle").font(.title2.weight(.bold)).foregroundStyle(.white).frame(width: 46, height: 46).background(.black, in: Circle())
                VStack(alignment: .leading) { Text("LocalMind").font(.headline); Text("Private workspace").font(.caption).foregroundStyle(.secondary) }
                Spacer()
                Button(action: close) { Image(systemName: "chevron.left").frame(width: 42, height: 42).background(.thinMaterial, in: Circle()) }
            }.padding(.horizontal, 22).padding(.top, 58).padding(.bottom, 18)
            Button { store.newChat(); close() } label: { Label("New chat", systemImage: "square.and.pencil").font(.headline).frame(maxWidth: .infinity).padding(.vertical, 14).foregroundStyle(.white).background(.black, in: RoundedRectangle(cornerRadius: 18, style: .continuous)) }.padding(.horizontal, 22)
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 2) {
                    Text("Workspace").font(.headline).padding(.top, 26).padding(.bottom, 8)
                    shortcut("Library", icon: "books.vertical.fill") { selectedWorkspace = "Library"; page = .library; close() }
                    shortcut("Projects", icon: "folder.fill") { selectedWorkspace = "Projects"; page = .chat; close() }
                    shortcut("Plugins", icon: "puzzlepiece.extension.fill") { selectedWorkspace = "Plugins"; page = .settings; close() }
                    shortcut("Scheduled", icon: "clock") { selectedWorkspace = "Scheduled"; page = .chat; close() }
                    shortcut("Remote", icon: "desktopcomputer") { selectedWorkspace = "Remote"; page = .settings; close() }
                    shortcut("Images", icon: "paintbrush.pointed.fill") { selectedWorkspace = "Images"; page = .library; close() }
                    Text("Recents").font(.headline).padding(.top, 28).padding(.bottom, 8)
                    ForEach(store.sessions) { session in
                        Button { Task { await store.select(session); close() } } label: {
                            VStack(alignment: .leading, spacing: 4) { Text(session.title).lineLimit(1); Text("Local conversation").font(.caption).foregroundStyle(.secondary) }
                                .frame(maxWidth: .infinity, alignment: .leading).padding(.vertical, 11).padding(.horizontal, 10)
                        }.buttonStyle(.plain).background(store.selectedSession?.id == session.id ? Color.primary.opacity(0.08) : .clear, in: RoundedRectangle(cornerRadius: 12))
                    }
                }.padding(.horizontal, 22)
            }
            HStack(spacing: 10) {
                Label("Search", systemImage: "magnifyingglass").foregroundStyle(.secondary).frame(maxWidth: .infinity, alignment: .leading).padding(.horizontal, 16).frame(height: 50).background(.thinMaterial, in: Capsule())
                Button { page = .settings; close() } label: { Image(systemName: "gearshape.fill").frame(width: 50, height: 50).background(.thinMaterial, in: Circle()) }
                Button { store.newChat(); close() } label: { Image(systemName: "square.and.pencil").frame(width: 50, height: 50).background(.thinMaterial, in: Circle()) }
            }.padding(20)
        }
    }

    private func shortcut(_ title: String, icon: String, action: @escaping () -> Void) -> some View {
        Button(action: action) { Label(title, systemImage: icon).frame(maxWidth: .infinity, alignment: .leading).padding(.vertical, 9).padding(.horizontal, 10) }
            .buttonStyle(.plain)
            .background(selectedWorkspace == title ? Color.primary.opacity(0.12) : .clear, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
    }
}

private struct ModernChat: View {
    @Bindable var store: WorkspaceStore
    @Binding var page: ModernPage
    @State private var prompt = ""
    @State private var attachment: ChatAttachment?
    @State private var isAttachmentMenuPresented = false
    @State private var isFileImporterPresented = false
    @State private var isCameraPresented = false
    @State private var isPhotoPickerPresented = false
    @State private var selectedPhotoItem: PhotosPickerItem?
    @FocusState private var isPromptFocused: Bool

    var body: some View {
        ZStack(alignment: .bottom) {
            LinearGradient(colors: [Color.indigo.opacity(0.16), Color.cyan.opacity(0.07), Color.clear], startPoint: .topLeading, endPoint: .bottomTrailing).ignoresSafeArea()
            if store.messages.isEmpty { ModernWelcome(page: $page) } else { messages }
            VStack(alignment: .leading, spacing: 12) {
                if isAttachmentMenuPresented {
                    AttachmentMenu(camera: { isAttachmentMenuPresented = false; isCameraPresented = true }, media: { isAttachmentMenuPresented = false; isPhotoPickerPresented = true }, files: { isAttachmentMenuPresented = false; isFileImporterPresented = true })
                        .transition(.move(edge: .bottom).combined(with: .opacity))
                }
                ModernComposer(prompt: $prompt, attachment: $attachment, isGenerating: store.isGenerating, mode: $store.generationMode, showAttachmentMenu: $isAttachmentMenuPresented, isPromptFocused: $isPromptFocused, send: { isPromptFocused = false; Task { await store.send(prompt: prompt, attachment: attachment); prompt = ""; attachment = nil } }, stop: store.stopGeneration)
            }
            .padding(.bottom, 5)
        }
        // Tapping empty conversation space dismisses the keyboard without changing tabs.
        .simultaneousGesture(TapGesture().onEnded { isPromptFocused = false })
        // PhotosPicker opens the actual system Photos library (including Videos), unlike Files.
        .photosPicker(isPresented: $isPhotoPickerPresented, selection: $selectedPhotoItem, matching: .any(of: [.images, .videos]))
        .onChange(of: selectedPhotoItem) { _, item in
            guard let item else { return }
            Task { await importPhotoLibraryItem(item) }
        }
        .fileImporter(isPresented: $isFileImporterPresented, allowedContentTypes: [.item], allowsMultipleSelection: false, onCompletion: importAttachment)
        .sheet(isPresented: $isCameraPresented) { CameraCapture { result in if case let .success(capturedAttachment) = result { attachment = capturedAttachment } else if case let .failure(error) = result { store.errorMessage = error.localizedDescription; store.isShowingError = true } } }
    }

    private var messages: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(spacing: 16) {
                    ForEach(store.messages) { message in
                        HStack {
                            if message.role == .user { Spacer() }
                            VStack(alignment: .leading, spacing: 8) {
                                if let fileName = message.fileName { MessageAttachmentPreview(message: message, fileName: fileName) }
                                if !message.content.isEmpty { Text(message.content) }
                                if message.content.isEmpty && message.fileName == nil { Text("…") }
                            }
                            .padding(13)
                            .background(message.role == .user ? Color.black : Color.primary.opacity(0.08), in: RoundedRectangle(cornerRadius: 18, style: .continuous))
                            .foregroundStyle(message.role == .user ? .white : .primary)
                            if message.role != .user { Spacer() }
                        }
                    }
                    Color.clear.frame(height: 142).id("bottom")
                }.padding(.horizontal, 18).padding(.top, 20)
            }.onChange(of: store.messages.count) { _, _ in withAnimation { proxy.scrollTo("bottom", anchor: .bottom) } }
        }
    }

    private func importAttachment(_ result: Result<[URL], Error>) {
        guard case let .success(urls) = result, let url = urls.first else { return }
        do { attachment = try ChatAttachment.cache(url) }
        catch { store.errorMessage = "Could not import \(url.lastPathComponent)."; store.isShowingError = true }
    }

    private func importPhotoLibraryItem(_ item: PhotosPickerItem) async {
        do {
            if item.supportedContentTypes.contains(where: { $0.conforms(to: .movie) }), let video = try await item.loadTransferable(type: PickedVideo.self) {
                attachment = try ChatAttachment.cache(video.url)
            } else if let imageData = try await item.loadTransferable(type: Data.self) {
                let imageType = item.supportedContentTypes.first(where: { $0.conforms(to: .image) })
                let fileExtension = imageType?.preferredFilenameExtension ?? "jpg"
                let temporaryURL = FileManager.default.temporaryDirectory.appendingPathComponent("photo-library-\(UUID().uuidString).\(fileExtension)")
                try imageData.write(to: temporaryURL, options: .atomic)
                attachment = try ChatAttachment.cache(temporaryURL)
            }
        } catch {
            store.errorMessage = "Could not load the selected photo or video."; store.isShowingError = true
        }
        selectedPhotoItem = nil
    }
}

/// Previews cached local media directly in the chat; older server history still has a useful file chip.
private struct MessageAttachmentPreview: View {
    let message: ChatMessage
    let fileName: String
    var body: some View {
        if let path = message.attachmentPath, let kind = message.attachmentKind {
            switch kind {
            case .image:
                if let image = UIImage(contentsOfFile: path) {
                    Image(uiImage: image).resizable().scaledToFit().frame(maxWidth: 220, maxHeight: 220).clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                } else { documentChip }
            case .video:
                VideoPlayer(player: AVPlayer(url: URL(fileURLWithPath: path))).frame(width: 220, height: 150).clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
            case .document:
                documentChip
            }
        } else {
            documentChip
        }
    }
    private var documentChip: some View { Label(fileName, systemImage: message.attachmentKind == .video ? "video.fill" : "doc.fill").font(.caption.weight(.medium)).lineLimit(1).padding(10).background(.white.opacity(0.16), in: RoundedRectangle(cornerRadius: 10, style: .continuous)) }
}

/// Composer-anchored action menu, matching the familiar chat attachment flow.
private struct AttachmentMenu: View {
    let camera: () -> Void; let media: () -> Void; let files: () -> Void
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            menuRow("Camera", icon: "camera.fill", action: camera)
            menuRow("Photo or Video", icon: "photo.on.rectangle.angled", action: media)
            menuRow("Files", icon: "doc", action: files)
        }
        .padding(8)
        .frame(width: 242)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 26, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 26, style: .continuous).strokeBorder(.primary.opacity(0.12)))
        .shadow(color: .black.opacity(0.22), radius: 18, y: 8)
        .padding(.leading, 24)
    }
    private func menuRow(_ title: String, icon: String, action: @escaping () -> Void) -> some View { Button(action: action) { Label(title, systemImage: icon).font(.headline).frame(maxWidth: .infinity, alignment: .leading).padding(.vertical, 12).padding(.horizontal, 10).contentShape(Rectangle()) }.buttonStyle(.plain) }
}

private struct ModernWelcome: View {
    @Binding var page: ModernPage
    var body: some View {
        ScrollView { VStack(spacing: 20) {
            Image(systemName: "sparkle").font(.system(size: 28, weight: .bold)).foregroundStyle(.white).frame(width: 68, height: 68).background(.black, in: Circle()).padding(.top, 80)
            Text("Ask anything").font(.system(size: 34, weight: .bold, design: .rounded))
            Text("Your private, local AI workspace").foregroundStyle(.secondary)
            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 12) {
                card("Deep research", icon: "magnifyingglass"); card("Analyze files", icon: "doc.text.magnifyingglass"); card("Create images", icon: "photo.badge.plus"); card("Brainstorm", icon: "lightbulb")
            }.padding(.top, 10)
        }.padding(.horizontal, 22).padding(.bottom, 150) }
    }
    private func card(_ title: String, icon: String) -> some View { Button { if title == "Create images" { page = .library } } label: { VStack(alignment: .leading, spacing: 12) { Image(systemName: icon).font(.title3); Text(title).font(.subheadline.weight(.semibold)); Text("Start a task").font(.caption).foregroundStyle(.secondary) }.frame(maxWidth: .infinity, minHeight: 98, alignment: .leading).padding(14).background(Color.primary.opacity(0.07), in: RoundedRectangle(cornerRadius: 20, style: .continuous)) }.buttonStyle(.plain) }
}

/// Material overlay that intentionally sits above the scroll content so it blurs content behind it.
private struct ModernComposer: View {
    @Binding var prompt: String; @Binding var attachment: ChatAttachment?; let isGenerating: Bool; @Binding var mode: GenerationMode; @Binding var showAttachmentMenu: Bool; var isPromptFocused: FocusState<Bool>.Binding; let send: () -> Void; let stop: () -> Void
    var body: some View { HStack(spacing: 10) {
        Button { isPromptFocused.wrappedValue = false; showAttachmentMenu = true } label: { Image(systemName: "plus").frame(width: 38, height: 38).background(Color.primary.opacity(0.09), in: Circle()) }
        TextField("Ask anything", text: $prompt, axis: .vertical).lineLimit(1...3).focused(isPromptFocused)
        if let attachment { DraftAttachmentChip(attachment: attachment) { self.attachment = nil } }
        Button { mode = mode == .flash ? .thinking : .flash } label: { Text(mode == .flash ? "Fast" : "Think").font(.caption.weight(.semibold)) }
        Button(action: isGenerating ? stop : send) { Image(systemName: isGenerating ? "stop.fill" : "arrow.up").frame(width: 38, height: 38).foregroundStyle(.white).background(isGenerating ? Color.red : Color.black, in: Circle()) }.disabled(!isGenerating && prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && attachment == nil)
    }.padding(8).background(.ultraThinMaterial, in: Capsule()).overlay(Capsule().strokeBorder(.primary.opacity(0.13))).shadow(color: .black.opacity(0.14), radius: 14, y: 5).padding(.horizontal, 18) }
}

/// Shows the selected upload before sending, the same way modern chat apps retain photo context.
private struct DraftAttachmentChip: View {
    let attachment: ChatAttachment
    let remove: () -> Void
    var body: some View {
        HStack(spacing: 5) {
            if attachment.kind == .image, let image = UIImage(contentsOfFile: attachment.url.path) {
                Image(uiImage: image).resizable().scaledToFill().frame(width: 25, height: 25).clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
            } else {
                Image(systemName: attachment.kind == .video ? "video.fill" : "doc.fill")
            }
            Text(attachment.name).lineLimit(1)
            Button(action: remove) { Image(systemName: "xmark.circle.fill") }
        }
        .font(.caption)
        .padding(.horizontal, 7).padding(.vertical, 5)
        .background(.primary.opacity(0.10), in: Capsule())
    }
}

/// UIKit camera bridge. The captured result is immediately copied to LocalMind's attachment cache.
private struct CameraCapture: UIViewControllerRepresentable {
    let completion: (Result<ChatAttachment, Error>) -> Void
    func makeCoordinator() -> Coordinator { Coordinator(completion: completion) }
    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.sourceType = .camera
        picker.mediaTypes = [UTType.image.identifier, UTType.movie.identifier]
        picker.delegate = context.coordinator
        return picker
    }
    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}

    final class Coordinator: NSObject, UINavigationControllerDelegate, UIImagePickerControllerDelegate {
        let completion: (Result<ChatAttachment, Error>) -> Void
        init(completion: @escaping (Result<ChatAttachment, Error>) -> Void) { self.completion = completion }
        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) { picker.dismiss(animated: true) }
        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]) {
            do {
                let attachment: ChatAttachment
                if let url = info[.mediaURL] as? URL {
                    attachment = try ChatAttachment.cache(url)
                } else if let image = info[.originalImage] as? UIImage, let data = image.jpegData(compressionQuality: 0.9) {
                    let temporaryURL = FileManager.default.temporaryDirectory.appendingPathComponent("camera-\(UUID().uuidString).jpg")
                    try data.write(to: temporaryURL, options: .atomic)
                    attachment = try ChatAttachment.cache(temporaryURL)
                } else { throw CameraError.unavailable }
                picker.dismiss(animated: true) { self.completion(.success(attachment)) }
            } catch { picker.dismiss(animated: true) { self.completion(.failure(error)) } }
        }
    }
    enum CameraError: LocalizedError { case unavailable; var errorDescription: String? { "The camera did not return a usable photo or video." } }
}

/// File-backed Photos transfer avoids loading large videos into memory before they are uploaded.
private struct PickedVideo: Transferable {
    let url: URL
    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(importedContentType: .movie) { received in
            let destination = FileManager.default.temporaryDirectory.appendingPathComponent("photo-library-video-\(UUID().uuidString).mov")
            try FileManager.default.copyItem(at: received.file, to: destination)
            return PickedVideo(url: destination)
        }
    }
}

private struct ModernBottomNavigation: View {
    @Binding var page: ModernPage
    @Namespace private var selectionNamespace
    var body: some View {
        // A conventional, stable tab bar: every destination is visible and tap targets do not resize.
        HStack(spacing: 0) {
            tab(.chat, "Home", "house.fill")
            tab(.library, "Library", "square.grid.2x2.fill")
            tab(.settings, "Settings", "gearshape.fill")
        }
        .padding(5)
        .frame(width: 274, height: 58)
        .background(.regularMaterial, in: Capsule())
        .overlay(Capsule().strokeBorder(.primary.opacity(0.10)))
        .shadow(color: .black.opacity(0.12), radius: 12, y: 5)
        .sensoryFeedback(.selection, trigger: page)
    }
    
    private func tab(_ item: ModernPage, _ title: String, _ icon: String) -> some View { Button { withAnimation(.spring(response: 0.30, dampingFraction: 0.82)) { page = item } } label: { ZStack { if page == item { Capsule().fill(.primary.opacity(0.16)).matchedGeometryEffect(id: "selected", in: selectionNamespace) }; VStack(spacing: 3) { Image(systemName: icon).frame(height: 26); Text(title).font(.caption2.weight(.semibold)) } }.frame(maxWidth: .infinity, maxHeight: .infinity).foregroundStyle(page == item ? .primary : .secondary) }.buttonStyle(.plain) }
}

private struct ModernLibrary: View { @Bindable var store: WorkspaceStore; var body: some View { ContentUnavailableView("Your library", systemImage: "square.grid.2x2", description: Text("Images and files created by your local workspace appear here.")) } }
private struct ModernSettings: View { @Binding var appearance: String; @Bindable var store: WorkspaceStore; var body: some View { Form { Section("Appearance") { Picker("Theme", selection: $appearance) { Text("System").tag("system"); Text("Light").tag("light"); Text("Dark").tag("dark") }.pickerStyle(.segmented) }; Section("Local runtime") { TextField("API URL", text: $store.apiBaseURL).textInputAutocapitalization(.never).autocorrectionDisabled() } } } }
