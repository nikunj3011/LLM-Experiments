//
//  ContentView.swift
//  LocalMind
//
//  Created by Nikunj Rathod on 2026-08-14.
//

import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    /// Owns the observable workspace for the lifetime of this scene.
    @State private var store = WorkspaceStore()
    /// Controls the native full-screen menu presented from the leading toolbar item.
    @State private var isMenuPresented = false
    /// Persisted appearance preference. `system` deliberately leaves the system choice intact.
    @AppStorage("appearance") private var appearance = "system"

    var body: some View {
        ZStack(alignment: .leading) {
            NavigationStack {
                // Keep routing here so individual features don't need to know about the app shell.
                Group {
                    switch store.destination {
                    case .chat: ChatView(store: store)
                    case .image, .video, .audio: StudioView(kind: store.destination, store: store)
                    case .gallery: GalleryView(store: store)
                    case .settings: SettingsView(store: store, appearance: $appearance)
                    }
                }
                .toolbar {
                    ToolbarItem(placement: .topBarLeading) {
                        Button("Menu", systemImage: "line.3.horizontal") { withAnimation(.snappy) { isMenuPresented = true } }
                    }
                }
                // The navigation stays visible above the app's persistent composer/navigation area.
                .safeAreaInset(edge: .bottom) { ExpandableBottomBar(store: store) }
            }
            // Edge swipe opens the menu without moving or resizing the live conversation.
            .simultaneousGesture(DragGesture(minimumDistance: 18).onEnded { value in
                guard value.startLocation.x < 30, value.translation.width > 85 else { return }
                withAnimation(.smooth(duration: 0.28)) { isMenuPresented = true }
            })

            if isMenuPresented {
                Color.black.opacity(0.32).ignoresSafeArea()
                    .onTapGesture { withAnimation(.smooth(duration: 0.24)) { isMenuPresented = false } }
                    .transition(.opacity)
                GeometryReader { proxy in
                    let drawerWidth = min(proxy.size.width * 0.84, 420)
                    WorkspaceSidebar(store: store, dismiss: { withAnimation(.smooth(duration: 0.24)) { isMenuPresented = false } })
                        .frame(width: drawerWidth)
                        .frame(maxHeight: .infinity)
                        .clipShape(UnevenRoundedRectangle(bottomTrailingRadius: 30, topTrailingRadius: 30))
                        .shadow(color: .black.opacity(0.20), radius: 16, x: 6)
                        .transition(.move(edge: .leading))
                }
            }
        }
        .tint(.primary)
        .preferredColorScheme(colorScheme)
        .task { await store.refresh() }
        .alert("LocalMind", isPresented: $store.isShowingError) { Button("OK", role: .cancel) {} } message: { Text(store.errorMessage) }
    }

    private var colorScheme: ColorScheme? { appearance == "light" ? .light : appearance == "dark" ? .dark : nil }
}

private struct WorkspaceSidebar: View {
    @Bindable var store: WorkspaceStore
    let dismiss: () -> Void
    var body: some View {
        VStack(spacing: 0) {
            HStack { Text("LocalMind").font(.title2.bold()); Spacer(); Button("Search", systemImage: "magnifyingglass") {}.buttonStyle(.bordered).clipShape(Circle()) }
                .padding(.horizontal, 26).padding(.top, 62).padding(.bottom, 22)
            ScrollView {
                VStack(alignment: .leading, spacing: 8) {
                    drawerButton("Image studio", icon: "paintbrush.pointed.fill", destination: .image)
                    drawerButton("Library", icon: "books.vertical.fill", destination: .gallery)
                    drawerButton("Projects", icon: "folder.fill", destination: .chat)
                    drawerButton("Remote", icon: "desktopcomputer", destination: .settings)
                    drawerButton("Plugins", icon: "puzzlepiece.extension.fill", destination: .settings)
                    Text("Recents").font(.title3.bold()).padding(.top, 34).padding(.bottom, 10)
                    ForEach(store.sessions) { session in
                        Button { Task { await store.select(session); dismiss() } } label: { Text(session.title).lineLimit(1).frame(maxWidth: .infinity, alignment: .leading).padding(.vertical, 10) }.buttonStyle(.plain)
                    }
                }.padding(.horizontal, 26)
            }
            HStack(spacing: 14) {
                Button { store.newChat(); dismiss() } label: { Label("Chat", systemImage: "square.and.pencil").font(.headline).foregroundStyle(.white).padding(.horizontal, 22).padding(.vertical, 14).background(.blue.gradient, in: Capsule()) }
                Spacer()
                Button { store.destination = .settings; dismiss() } label: { Image(systemName: "gearshape.fill").font(.title3).frame(width: 52, height: 52).background(.primary.opacity(0.10), in: Circle()) }
            }.padding(22)
        }
        // The drawer respects the app's system/light/dark preference rather than forcing black.
        .foregroundStyle(.primary)
        .background(.background)
    }

    private func drawerButton(_ label: String, icon: String, destination: AppDestination) -> some View {
        Button { store.destination = destination; dismiss() } label: { Label(label, systemImage: icon).font(.title3.weight(.medium)).frame(maxWidth: .infinity, alignment: .leading).padding(.vertical, 10) }.buttonStyle(.plain)
    }
}

private struct ExpandableBottomBar: View {
    @Bindable var store: WorkspaceStore
    /// The center button exposes infrequent actions without sacrificing tab-bar access.
    @State private var isExpanded = false
    @State private var isQuickActionTrayVisible = false
    @State private var highlightedQuickAction: QuickAction?

    var body: some View {
        VStack(spacing: 8) {
            if isQuickActionTrayVisible {
                quickActionTray
                    .transition(.scale(scale: 0.8, anchor: .bottom).combined(with: .opacity))
            }
            if isExpanded {
                HStack(spacing: 10) {
                    actionButton("New chat", icon: "square.and.pencil") { store.newChat() }
                    actionButton("Studios", icon: "sparkles") { store.destination = .image }
                    actionButton("Gallery", icon: "square.grid.2x2") { store.destination = .gallery }
                }.padding(.horizontal, 7).padding(.vertical, 7).background(.regularMaterial, in: Capsule())
                    .transition(.move(edge: .bottom).combined(with: .opacity))
            }
            // Every item gets flexible width so the bar scales smoothly across iPhone sizes.
            HStack(spacing: 3) {
                tabButton(.chat, icon: "house.fill")
                tabButton(.image, icon: "play.rectangle.fill")
                Image(systemName: isExpanded ? "xmark" : "plus")
                    .font(.headline.weight(.bold))
                    .frame(maxWidth: .infinity, minHeight: 34)
                    .foregroundStyle(.primary)
                    .background(.background, in: Circle())
                    .overlay(Circle().strokeBorder(.primary.opacity(0.12)))
                    .symbolEffect(.bounce, value: isExpanded)
                    .contentShape(Circle())
                    .onTapGesture { guard !isQuickActionTrayVisible else { return }; withAnimation(.spring(response: 0.35, dampingFraction: 0.78)) { isExpanded.toggle() } }
                    .gesture(quickActionGesture)
                    .accessibilityLabel(isExpanded ? "Close actions" : "Open actions")
                tabButton(.gallery, icon: "safari.fill")
                tabButton(.settings, icon: "person.crop.circle.fill")
                }
            .padding(5)
            .frame(width: 280)
            .background(.regularMaterial, in: Capsule())
            .overlay(Capsule().strokeBorder(.primary.opacity(0.08)))
            .shadow(color: .black.opacity(0.12), radius: 10, y: 4)
        }
        .padding(.bottom, 5)
        .frame(maxWidth: .infinity)
        // Native haptics confirm the expand/collapse state without custom UIKit code.
        .sensoryFeedback(.selection, trigger: isExpanded)
        .sensoryFeedback(.selection, trigger: highlightedQuickAction)
    }

    /// Icon-only tabs preserve the compact, familiar social-app navigation silhouette.
    private func tabButton(_ destination: AppDestination, icon: String) -> some View {
        Button { withAnimation(.spring(response: 0.38, dampingFraction: 0.74)) { store.destination = destination; isExpanded = false } } label: {
            Image(systemName: icon).font(.title3.weight(.semibold))
                .frame(maxWidth: .infinity, minHeight: 34)
                .foregroundStyle(store.destination == destination ? .primary : .secondary)
                .background(store.destination == destination ? Color.primary.opacity(0.13) : Color.clear, in: Capsule())
                .scaleEffect(store.destination == destination ? 1.08 : 0.92)
                .animation(.spring(response: 0.28, dampingFraction: 0.72), value: store.destination)
        }
    }

    private func actionButton(_ title: String, icon: String, action: @escaping () -> Void) -> some View {
        Button { action(); withAnimation { isExpanded = false } } label: {
            Label(title, systemImage: icon).font(.caption.weight(.semibold)).padding(.horizontal, 10).padding(.vertical, 8)
        }.buttonStyle(.plain)
    }

    /// Custom long-press-and-drag interaction: hold the add button, slide over an action, then release.
    private var quickActionGesture: some Gesture {
        LongPressGesture(minimumDuration: 0.28)
            .sequenced(before: DragGesture(minimumDistance: 0))
            .onChanged { value in
                guard case .second(true, let drag?) = value else { return }
                withAnimation(.snappy) { isQuickActionTrayVisible = true; isExpanded = false }
                highlightedQuickAction = quickAction(for: drag.translation.width)
            }
            .onEnded { value in
                if case .second(true, let drag?) = value {
                    perform(quickAction(for: drag.translation.width))
                }
                withAnimation(.snappy) { isQuickActionTrayVisible = false; highlightedQuickAction = nil }
            }
    }

    private var quickActionTray: some View {
        HStack(spacing: 12) {
            ForEach(QuickAction.allCases) { action in
                VStack(spacing: 6) {
                    Image(systemName: action.icon).font(.headline)
                    Text(action.title).font(.caption2.weight(.semibold))
                }
                .frame(width: 64, height: 54)
                .foregroundStyle(highlightedQuickAction == action ? .white : .primary)
                .background(highlightedQuickAction == action ? Color.blue : Color.clear, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
            }
        }
        .padding(8)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 24, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 24, style: .continuous).strokeBorder(.primary.opacity(0.08)))
    }

    private func quickAction(for translation: CGFloat) -> QuickAction? {
        switch translation {
        case ..<(-44): .newChat
        case 44...: .gallery
        default: .image
        }
    }

    private func perform(_ action: QuickAction?) {
        switch action {
        case .newChat: store.newChat()
        case .image: store.destination = .image
        case .gallery: store.destination = .gallery
        case nil: break
        }
    }
}

private enum QuickAction: CaseIterable, Identifiable {
    case newChat, image, gallery
    var id: Self { self }
    var title: String { switch self { case .newChat: "Chat"; case .image: "Image"; case .gallery: "Gallery" } }
    var icon: String { switch self { case .newChat: "square.and.pencil"; case .image: "photo"; case .gallery: "square.grid.2x2" } }
}

private struct ChatView: View {
    @Bindable var store: WorkspaceStore
    @State private var prompt = ""; @State private var attachment: ChatAttachment?; @State private var showImporter = false
    var body: some View {
        VStack(spacing: 0) { if store.messages.isEmpty { EmptyConversationView(destination: $store.destination) } else { conversation }; Composer(prompt: $prompt, attachment: $attachment, isGenerating: store.isGenerating, mode: $store.generationMode, showImporter: $showImporter, send: { Task { await store.send(prompt: prompt, attachment: attachment); prompt = ""; attachment = nil } }, stop: { store.stopGeneration() }) }
        .background(Color(uiColor: .systemBackground))
        .navigationTitle(store.selectedSession?.title ?? "New conversation")
        .toolbar { ToolbarItem(placement: .topBarTrailing) { Menu { Picker("Active model", selection: $store.selectedModelID) { ForEach(store.models) { Text($0.name).tag($0.id) } } } label: { Label(store.activeModel?.name ?? "Select model", systemImage: "cpu") } } }
        .fileImporter(isPresented: $showImporter, allowedContentTypes: [.item], allowsMultipleSelection: false) { result in guard case let .success(urls) = result, let url = urls.first else { return }; attachment = ChatAttachment(url: url, name: url.lastPathComponent) }
    }
    private var conversation: some View { ScrollViewReader { proxy in ScrollView { LazyVStack(spacing: 18) { ForEach(store.messages) { message in MessageBubble(message: message) }; if store.isGenerating { HStack(spacing: 7) { ProgressView().controlSize(.small); Text("LocalMind is thinking…").foregroundStyle(.secondary) }.frame(maxWidth: .infinity, alignment: .leading).padding(.horizontal) }; Color.clear.frame(height: 1).id("bottom") }.padding(.horizontal).padding(.vertical, 24) }.onChange(of: store.messages.count) { _, _ in withAnimation { proxy.scrollTo("bottom", anchor: .bottom) } } } }
}

private struct EmptyConversationView: View {
    @Binding var destination: AppDestination
    var body: some View {
        ScrollView {
            VStack(spacing: 22) {
                Image(systemName: "sparkle").font(.system(size: 31, weight: .bold)).foregroundStyle(.white).frame(width: 70, height: 70).background(.black, in: Circle()).padding(.top, 58)
                Text("Ask anything").font(.system(size: 34, weight: .bold, design: .rounded))
                Text("Your private, local AI workspace").foregroundStyle(.secondary)
                LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 11) {
                    feature("Deep research", icon: "magnifyingglass", prompt: "Help me research a topic")
                    feature("Analyze files", icon: "doc.text.magnifyingglass", prompt: "Analyze this document")
                    feature("Create images", icon: "photo.badge.plus", prompt: "Create an image")
                    feature("Brainstorm", icon: "lightbulb", prompt: "Help me brainstorm")
                }.padding(.top, 8)
                HStack(spacing: 8) { Label("Private", systemImage: "lock.fill"); Label("Local runtime", systemImage: "bolt.fill") }.font(.caption.weight(.medium)).foregroundStyle(.secondary)
            }.padding(.horizontal, 22)
        }.frame(maxWidth: .infinity, maxHeight: .infinity)
    }
    private func feature(_ title: String, icon: String, prompt: String) -> some View { Button { if title == "Create images" { destination = .image } } label: { VStack(alignment: .leading, spacing: 12) { Image(systemName: icon).font(.title3); Text(title).font(.subheadline.weight(.semibold)); Text(prompt).font(.caption).foregroundStyle(.secondary).lineLimit(2) }.frame(maxWidth: .infinity, minHeight: 104, alignment: .leading).padding(14).background(.quaternary, in: RoundedRectangle(cornerRadius: 18)) }.buttonStyle(.plain) }
}
private struct MessageBubble: View { let message: ChatMessage; var body: some View { HStack(alignment: .top, spacing: 10) { if message.role == .assistant { Image(systemName: "sparkle").frame(width: 34, height: 34).background(.black, in: RoundedRectangle(cornerRadius: 11)).foregroundStyle(.white) }; VStack(alignment: message.role == .user ? .trailing : .leading, spacing: 7) { if let name = message.fileName { Label(name, systemImage: "paperclip").font(.caption).foregroundStyle(.secondary) }; Text(message.content.isEmpty ? "…" : message.content).textSelection(.enabled).padding(13).background(message.role == .user ? AnyShapeStyle(Color.black) : AnyShapeStyle(.background), in: RoundedRectangle(cornerRadius: 18)).foregroundStyle(message.role == .user ? .white : .primary) }.frame(maxWidth: .infinity, alignment: message.role == .user ? .trailing : .leading); if message.role == .user { Image(systemName: "person.fill").frame(width: 34, height: 34).background(.quaternary, in: RoundedRectangle(cornerRadius: 11)).foregroundStyle(.secondary) } } } }
private struct Composer: View { @Binding var prompt: String; @Binding var attachment: ChatAttachment?; let isGenerating: Bool; @Binding var mode: GenerationMode; @Binding var showImporter: Bool; let send: () -> Void; let stop: () -> Void; var body: some View { VStack(spacing: 8) { if let attachment { HStack { Label(attachment.name, systemImage: "doc.fill").lineLimit(1); Spacer(); Button("Remove", systemImage: "xmark.circle.fill") { self.attachment = nil }.labelStyle(.iconOnly) }.font(.caption).padding(.horizontal) }; HStack(alignment: .bottom, spacing: 10) { Button { showImporter = true } label: { Image(systemName: "plus") }.buttonStyle(.bordered); TextField(mode == .thinking ? "Ask anything, think deeply…" : "Ask anything", text: $prompt, axis: .vertical).lineLimit(1...5).padding(11).background(.quaternary, in: RoundedRectangle(cornerRadius: 18)); if isGenerating { Button(action: stop) { Image(systemName: "stop.fill") }.buttonStyle(.borderedProminent).tint(.red) } else { Button(action: send) { Image(systemName: "arrow.up") }.buttonStyle(.borderedProminent).tint(.black).disabled(prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && attachment == nil) } }; HStack { Picker("Mode", selection: $mode) { ForEach(GenerationMode.allCases) { Text($0.title).tag($0) } }.pickerStyle(.segmented); Spacer(); Label("Files, photos & video", systemImage: "paperclip").font(.caption2).foregroundStyle(.secondary) }.padding(.leading, 44) }.padding(10).background(.regularMaterial, in: RoundedRectangle(cornerRadius: 26)).overlay(RoundedRectangle(cornerRadius: 26).strokeBorder(.primary.opacity(0.09))).shadow(color: .black.opacity(0.08), radius: 12, y: 4).padding(.horizontal, 14).padding(.top, 8).padding(.bottom, 4) } }
private struct StudioView: View { let kind: AppDestination; @Bindable var store: WorkspaceStore; @State private var direction = ""; var body: some View { Form { Section { TextEditor(text: $direction).frame(minHeight: 150) } header: { Text("Direction") } footer: { Text("This creates a new task in chat using your selected local model.") }; Section { Picker("Output", selection: .constant("Square")) { Text("Square").tag("Square"); Text("Widescreen").tag("Wide") }; Button("Create", systemImage: "sparkles") { store.destination = .chat; Task { await store.send(prompt: "Create a \(kind.title.lowercased()): \(direction)", attachment: nil) } }.disabled(direction.isEmpty) } }.navigationTitle(kind.title) } }
private struct GalleryView: View { @Bindable var store: WorkspaceStore; var body: some View { Group { if store.gallery.isEmpty { ContentUnavailableView("Nothing here yet", systemImage: "photo.on.rectangle", description: Text("Run a local creative workflow, then refresh.")) } else { ScrollView { LazyVGrid(columns: [GridItem(.adaptive(minimum: 145))], spacing: 14) { ForEach(store.gallery) { asset in AsyncImage(url: asset.url) { $0.resizable().scaledToFill() } placeholder: { ProgressView() }.frame(height: 145).clipShape(RoundedRectangle(cornerRadius: 16)) } }.padding() } } }.navigationTitle("Gallery").toolbar { Button("Refresh", systemImage: "arrow.clockwise") { Task { await store.refreshGallery() } } } } }
private struct SettingsView: View {
    @Bindable var store: WorkspaceStore
    @Binding var appearance: String
    var body: some View {
        Form {
            Section("Appearance") {
                Picker("Theme", selection: $appearance) {
                    Text("System").tag("system")
                    Text("Light").tag("light")
                    Text("Dark").tag("dark")
                }
                .pickerStyle(.segmented)
            }
            Section("Local runtime") {
                TextField("API base URL", text: $store.apiBaseURL).textInputAutocapitalization(.never).autocorrectionDisabled()
                Text("Use your Mac's LAN address on a physical iPhone, e.g. http://192.168.1.20:8000/api.").font(.footnote).foregroundStyle(.secondary)
            }
            Section("Agent behavior") { Toggle("Include context by default", isOn: .constant(true)); Toggle("Private workspace", isOn: .constant(true)) }
            Section("Connections") { LabeledContent("Local Qwen runtime", value: "Ready") }
        }.navigationTitle("Settings")
    }
}
#Preview { ContentView() }
