#pragma once

#include <signal.h>
#include <atomic>
#include <iostream>
#include <functional>

namespace lvmapping {

/**
 * @class SignalHandler
 * @brief Singleton class to handle system signals like SIGINT (Ctrl+C)
 */
class SignalHandler {
public:
    /**
     * @brief Get the instance of the signal handler
     * @return Reference to the singleton instance
     */
    static SignalHandler& getInstance() {
        static SignalHandler instance;
        return instance;
    }
    
    /**
     * @brief Initialize the signal handler
     * @param cleanup_callback Function to call during cleanup
     */
    void init(std::function<void()> cleanup_callback = nullptr) {
        cleanup_func_ = cleanup_callback;
        struct sigaction sigIntHandler;
        sigIntHandler.sa_handler = SignalHandler::signalHandler;
        sigemptyset(&sigIntHandler.sa_mask);
        sigIntHandler.sa_flags = 0;
        
        // Register the signal handler for SIGINT (Ctrl+C)
        sigaction(SIGINT, &sigIntHandler, NULL);
        
        // Also handle SIGTERM
        sigaction(SIGTERM, &sigIntHandler, NULL);
        
        std::cout << "Signal handler initialized. Press Ctrl+C to safely terminate..." << std::endl;
        
        // Reset the termination flag
        terminate_ = false;
    }
    
    /**
     * @brief Check if termination has been requested
     * @return True if termination was requested
     */
    bool shouldTerminate() const {
        return terminate_;
    }
    
    /**
     * @brief Set termination flag
     */
    void requestTermination() {
        if (!terminate_) {
            std::cout << "\nTermination requested. Cleaning up..." << std::endl;
            terminate_ = true;
        }
    }
    
    /**
     * @brief Run cleanup function if set
     */
    void cleanup() {
        if (cleanup_func_) {
            cleanup_func_();
        }
    }
    
private:
    SignalHandler() : terminate_(false), cleanup_func_(nullptr) {}
    ~SignalHandler() {}
    
    // Delete copy constructor and assignment operator
    SignalHandler(const SignalHandler&) = delete;
    SignalHandler& operator=(const SignalHandler&) = delete;
    
    // Static signal handler function
    static void signalHandler(int signal) {
        std::cout << "\nReceived signal " << signal << std::endl;
        SignalHandler::getInstance().requestTermination();
        
        // Only cleanup immediately for SIGTERM, as SIGINT will be handled in the main loop
        if (signal == SIGTERM) {
            SignalHandler::getInstance().cleanup();
        }
    }
    
    // Flag to indicate termination request
    std::atomic<bool> terminate_;
    
    // Cleanup function to call on termination
    std::function<void()> cleanup_func_;
};

} // namespace lvmapping
