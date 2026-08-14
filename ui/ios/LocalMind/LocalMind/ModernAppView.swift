import SwiftUI
import UniformTypeIdentifiers

/// The active application shell. It deliberately owns navigation and presentation state in one
/// small place, while `WorkspaceStore` continues to own the backend and conversation state.
struct ModernAppView: View {
    @State private var store = WorkspaceStore()
    @State private var page: ModernPage = .chat
    @State private var isDrawerOpen = false
    @State private var isNavExpanded = false
    @AppStorage("appearance") private var appearance = "system"

    var body: some View {
        ZStack(alignment: .leading) {
            NavigationStack {
                pageContent
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
            ModernChat(store: store, page: $page, isNavExpanded: $isNavExpanded)
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
                    shortcut("Image studio", icon: "paintbrush.pointed.fill") { page = .library; close() }
                    shortcut("Library", icon: "books.vertical.fill") { page = .library; close() }
                    shortcut("Projects", icon: "folder.fill") { page = .chat; close() }
                    shortcut("Remote", icon: "desktopcomputer") { page = .settings; close() }
                    shortcut("Plugins", icon: "puzzlepiece.extension.fill") { page = .settings; close() }
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
    }
}

private struct ModernChat: View {
    @Bindable var store: WorkspaceStore
    @Binding var page: ModernPage
    @Binding var isNavExpanded: Bool
    @State private var prompt = ""
    @State private var attachment: ChatAttachment?
    @State private var isImporterPresented = false

    var body: some View {
        ZStack(alignment: .bottom) {
            LinearGradient(colors: [Color.indigo.opacity(0.16), Color.cyan.opacity(0.07), Color.clear], startPoint: .topLeading, endPoint: .bottomTrailing).ignoresSafeArea()
            if store.messages.isEmpty { ModernWelcome(page: $page) } else { messages }
            VStack(spacing: 9) {
                ModernComposer(prompt: $prompt, attachment: $attachment, isGenerating: store.isGenerating, mode: $store.generationMode, showImporter: $isImporterPresented, send: { Task { await store.send(prompt: prompt, attachment: attachment); prompt = ""; attachment = nil } }, stop: store.stopGeneration)
                ModernBottomNavigation(page: $page, isExpanded: $isNavExpanded)
            }.padding(.bottom, 5)
        }
        .fileImporter(isPresented: $isImporterPresented, allowedContentTypes: [.item], allowsMultipleSelection: false) { result in
            guard case let .success(urls) = result, let url = urls.first else { return }
            attachment = ChatAttachment(url: url, name: url.lastPathComponent)
        }
    }

    private var messages: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(spacing: 16) {
                    ForEach(store.messages) { message in
                        HStack { if message.role == .user { Spacer() }; Text(message.content.isEmpty ? "…" : message.content).padding(13).background(message.role == .user ? Color.black : Color.primary.opacity(0.08), in: RoundedRectangle(cornerRadius: 18, style: .continuous)).foregroundStyle(message.role == .user ? .white : .primary); if message.role != .user { Spacer() } }
                    }
                    Color.clear.frame(height: 142).id("bottom")
                }.padding(.horizontal, 18).padding(.top, 20)
            }.onChange(of: store.messages.count) { _, _ in withAnimation { proxy.scrollTo("bottom", anchor: .bottom) } }
        }
    }
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
    @Binding var prompt: String; @Binding var attachment: ChatAttachment?; let isGenerating: Bool; @Binding var mode: GenerationMode; @Binding var showImporter: Bool; let send: () -> Void; let stop: () -> Void
    var body: some View { HStack(spacing: 10) {
        Button { showImporter = true } label: { Image(systemName: "plus").frame(width: 38, height: 38).background(Color.primary.opacity(0.09), in: Circle()) }
        TextField("Ask anything", text: $prompt, axis: .vertical).lineLimit(1...3)
        Button { mode = mode == .flash ? .thinking : .flash } label: { Text(mode == .flash ? "Fast" : "Think").font(.caption.weight(.semibold)) }
        Button(action: isGenerating ? stop : send) { Image(systemName: isGenerating ? "stop.fill" : "arrow.up").frame(width: 38, height: 38).foregroundStyle(.white).background(isGenerating ? Color.red : Color.black, in: Circle()) }.disabled(!isGenerating && prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && attachment == nil)
    }.padding(8).background(.ultraThinMaterial, in: Capsule()).overlay(Capsule().strokeBorder(.primary.opacity(0.13))).shadow(color: .black.opacity(0.14), radius: 14, y: 5).padding(.horizontal, 18) }
}

private struct ModernBottomNavigation: View {
    @Binding var page: ModernPage; @Binding var isExpanded: Bool
    @Namespace private var glassNamespace
    @Namespace private var selectionNamespace
    var body: some View {
        // A shared glass identity lets iOS morph the compact control into the full tab bar.
        // The spring is deliberately under-damped enough to feel responsive without growing the bar.
        GlassEffectContainer(spacing: 0) {
            if isExpanded {
                HStack(spacing: 0) {
                    tab(.chat, "Home", "house.fill")
                    tab(.library, "Library", "square.grid.2x2.fill")
                    tab(.settings, "Settings", "gearshape.fill")
                }
                .padding(5)
                .frame(width: 274, height: 54)
                .glassEffect(.regular.tint(.indigo.opacity(0.08)).interactive(), in: Capsule())
                .glassEffectID("bottom-navigation", in: glassNamespace)
                .transition(.scale(scale: 0.72, anchor: .leading).combined(with: .opacity))
            } else {
                HStack(spacing: 0) {
                    Button {
                        withAnimation(.spring(response: 0.42, dampingFraction: 0.76)) { isExpanded = true }
                    } label: {
                        Image(systemName: "house.fill").frame(width: 52, height: 52)
                    }
                    .buttonStyle(.plain)

                    Divider().frame(height: 22).opacity(0.35)

                    // Settings remains directly reachable without requiring expansion.
                    Button {
                        withAnimation(.spring(response: 0.32, dampingFraction: 0.82)) { page = .settings }
                    } label: {
                        Image(systemName: "gearshape.fill").frame(width: 52, height: 52)
                    }
                    .buttonStyle(.plain)
                }
                .padding(2)
                .frame(width: 108, height: 54)
                .glassEffect(.regular.tint(.indigo.opacity(0.08)).interactive(), in: Capsule())
                .glassEffectID("bottom-navigation", in: glassNamespace)
                .transition(.scale(scale: 0.72, anchor: .leading).combined(with: .opacity))
            }
        }
        .frame(maxWidth: .infinity, alignment: .center)
        .contentShape(Capsule())
        .sensoryFeedback(.selection, trigger: isExpanded)
    }
    
    private func tab(_ item: ModernPage, _ title: String, _ icon: String) -> some View { Button { withAnimation(.spring(response: 0.32, dampingFraction: 0.8)) { page = item; if item == .chat { isExpanded = false } } } label: { VStack(spacing: 3) { ZStack { if page == item { Capsule().fill(.primary.opacity(0.14)).matchedGeometryEffect(id: "selected", in: selectionNamespace) }; Image(systemName: icon) }.frame(height: 26); Text(title).font(.caption2.weight(.semibold)) }.frame(maxWidth: .infinity, maxHeight: .infinity).foregroundStyle(page == item ? .primary : .secondary) }.buttonStyle(.plain) }
}

private struct ModernLibrary: View { @Bindable var store: WorkspaceStore; var body: some View { ContentUnavailableView("Your library", systemImage: "square.grid.2x2", description: Text("Images and files created by your local workspace appear here.")) } }
private struct ModernSettings: View { @Binding var appearance: String; @Bindable var store: WorkspaceStore; var body: some View { Form { Section("Appearance") { Picker("Theme", selection: $appearance) { Text("System").tag("system"); Text("Light").tag("light"); Text("Dark").tag("dark") }.pickerStyle(.segmented) }; Section("Local runtime") { TextField("API URL", text: $store.apiBaseURL).textInputAutocapitalization(.never).autocorrectionDisabled() } } } }
