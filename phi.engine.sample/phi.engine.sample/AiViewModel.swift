//
//  AiViewModel.swift
//  phi.engine.sample
//
//  Created by Filip W on 15.04.2024.
//

import Foundation

class AiViewModel: ObservableObject {
    var engine: PhiEngine?
    let inferenceOptions: InferenceOptions = InferenceOptions(tokenCount: 100, temperature: 0.0, topP: 1.0, repeatPenalty: 1.0, repeatLastN: 64, seed: 299792458)
    @Published var isLoading: Bool = false
    @Published var isLoadingEngine: Bool = false
    @Published var messages: [ChatMessage] = []
    @Published var prompt: String = ""
    @Published var isReady: Bool = false
    
    func loadModel() async {
        DispatchQueue.main.async {
            self.isLoadingEngine = true
        }
        self.engine = PhiEngine(systemInstruction: nil)
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
                self.messages.append(ChatMessage(text: question, isUser: true))
            }
            
            let messageText = engine.runInference(promptText: question, inferenceOptions: self.inferenceOptions)
    
            DispatchQueue.main.async {
                self.messages.append(ChatMessage(text: messageText, isUser: false))
                self.isLoading = false
            }
        }
    }
}
