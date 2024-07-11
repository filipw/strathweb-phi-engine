//
//  AiViewModel.swift
//  phi.engine.sample
//
//  Created by Filip W on 15.04.2024.
//

import Foundation

class Phi3ViewModel: ObservableObject {
    var engine: PhiEngine?
    let inferenceOptions: InferenceOptions
    @Published var isLoading: Bool = false
    @Published var isLoadingEngine: Bool = false
    @Published var messages: [ChatMessage] = []
    @Published var prompt: String = ""
    @Published var isReady: Bool = false
    
    init() {
        let inferenceOptionsBuilder = InferenceOptionsBuilder()
        try! inferenceOptionsBuilder.withTemperature(temperature: 0.9)
        try! inferenceOptionsBuilder.withSeed(seed: 146628346)
        self.inferenceOptions = try! inferenceOptionsBuilder.build()
    }
    
    func loadModel() async {
        DispatchQueue.main.async {
            self.isLoadingEngine = true
        }
        
        let engineBuilder = PhiEngineBuilder()
        try! engineBuilder.withSystemInstruction(systemInstruction: "You are a hockey poet")
        try! engineBuilder.withEventHandler(eventHandler: BoxedPhiEventHandler(handler: ModelEventsHandler(parent: self)))
        
        self.engine = try! engineBuilder.build(cacheDir: FileManager.default.temporaryDirectory.path())
        DispatchQueue.main.async {
            self.isLoadingEngine = false
            self.isReady = true
        }
    }
    
    func fetchAIResponse() async {
        if let engine = self.engine {
            let question = self.prompt
            DispatchQueue.main.async {
                self.isLoading = true
                self.prompt = ""
                self.messages.append(ChatMessage(text: question, isUser: true, state: .ok))
                self.messages.append(ChatMessage(text: "", isUser: false, state: .waiting))
            }
            
            let inferenceResult = try! engine.runInference(promptText: question, inferenceOptions: self.inferenceOptions)
            print("\nTokens Generated: \(inferenceResult.tokenCount), Tokens per second: \(inferenceResult.tokensPerSecond), Duration: \(inferenceResult.duration)s")
            
            DispatchQueue.main.async {
                self.isLoading = false
            }
        }
    }
    
    class ModelEventsHandler : PhiEventHandler {
        unowned let parent: Phi3ViewModel
        
        init(parent: Phi3ViewModel) {
            self.parent = parent
        }
        
        func onInferenceToken(token: String) throws {
            DispatchQueue.main.async {
                if let lastMessage = self.parent.messages.last {
                    let updatedText = lastMessage.text + token
                    if let index = self.parent.messages.firstIndex(where: { $0.id == lastMessage.id }) {
                        self.parent.messages[index] = ChatMessage(text: updatedText, isUser: false, state: .ok)
                    }
                }
            }
        }
        
        func onModelLoaded() throws {
            print("MODEL LOADED")
        }
    }
}
