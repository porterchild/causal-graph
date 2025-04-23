import sys
from expert_system_librarian import ExpertSystemLibrarian # Import the new agent class
from llm_interface import client as llm_client # Still need to check client availability

# --- CLI Application Logic (Simplified) ---

def print_help():
    """Prints a user-friendly help message."""
    print("\nHow I can help:")
    print("  - State facts or rules: Just type them naturally (e.g., 'It might rain today', 'If it rains, the grass gets wet').")
    print("  - Ask 'what if' questions: Ask about probabilities (e.g., 'What's the chance of rain?', 'Will the alarm sound?').")
    print("  - Ask 'why' questions: Ask for likely causes (e.g., 'Why did the alarm ring?', 'What could cause the voltage sag?').")
    print("  - See the current model: Ask 'Show me the facts' or 'What are the rules?'.")
    print("  - Get help: Type 'help'.")
    print("  - Exit: Type 'quit' or 'exit'.")
    print("-" * 20)

def run_cli():
    """Runs the interactive command-line interface using the ExpertSystemLibrarian."""
    if not llm_client:
        print("Error: LLM client not initialized. Please ensure OPENROUTER_API_KEY is set.")
        print("The application cannot run without LLM connectivity.")
        sys.exit(1)

    # Initialize the librarian agent
    librarian = ExpertSystemLibrarian(debug=False) # Debug flag passed here

    print("Causal Graph ProbLog Agent (via Librarian)")
    print("Tell me facts, ask 'what if' (probability), or ask 'why' (causes). Type 'help' or 'quit'.")
    print("-" * 20)

    while True:
        try:
            user_input = input("> ").strip()

            if not user_input:
                continue

            # Process input using the librarian
            response_type, response_message = librarian.process_input(user_input)

            # Handle response from librarian
            if response_type == "quit":
                print(response_message)
                break
            elif response_type == "help":
                print_help()
            elif response_type == "response" or response_type == "error":
                print(response_message)
            # else: # Should not happen
            #     print(f"Unknown response type from librarian: {response_type}")

        except EOFError:
            print("\nExiting.")
            break
        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except Exception as e:
            print(f"\nAn unexpected error occurred in the CLI loop: {e}")
            # Optionally continue or break based on severity
            # break

if __name__ == "__main__":
    run_cli() # Run the simplified CLI
