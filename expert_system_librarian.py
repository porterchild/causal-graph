import json
from llm_interface import ProblogLLMInterface, client as llm_client, OpenAIError

class ExpertSystemLibrarian:
    """
    Acts as an agent to manage interaction with the ProblogLLMInterface
    based on natural language user input. It determines intent and calls
    the appropriate underlying methods.
    """
    def __init__(self, llm_model="google/gemini-2.5-flash-preview", debug=False):
        """
        Initializes the librarian and the underlying ProbLog interface.

        Args:
            llm_model (str): The LLM model identifier to use for intent recognition and translations.
            debug (bool): Enable debug printing in the ProbLog interface.
        """
        self.llm_model = llm_model
        self.interface = ProblogLLMInterface(llm_model=llm_model, debug=debug)
        if not llm_client:
            print("Warning: LLM client not initialized. Librarian functionality will be limited.")

    def _get_intent_and_payload(self, user_input: str) -> tuple[str, str | None]:
        """
        Uses LLM to determine the user's intent and extract the relevant payload.
        (Copied and adapted from the previous cli_app.py version)

        Args:
            user_input (str): The raw user input.

        Returns:
            tuple[str, str | None]: A tuple containing the intent (e.g., "ADD_FACT", "DEDUCTIVE_QUERY")
                                     and the payload (the statement/query/observation, or None).
                                     Returns ("UNKNOWN", user_input) on failure or unclear intent.
        """
        if not llm_client:
            print("Error: LLM client not available for intent recognition.")
            return "UNKNOWN", user_input

        # Simple keyword checks first
        lower_input = user_input.lower()
        if lower_input in ["quit", "exit", "bye"]:
            return "QUIT", None
        if lower_input == "help":
            return "HELP", None
        if lower_input in ["show facts", "list facts", "show model", "what are the facts?", "what are the rules?"]:
            return "SHOW_MODEL", None

        # LLM for classification
        prompt = f"""Analyze the user's request and classify its intent. Extract the core statement, question, or observation.
Possible intents are:
- ADD_FACT: User is stating a fact or rule to add to the knowledge base.
- DEDUCTIVE_QUERY: User is asking for the probability of an outcome ('what is the probability...', 'will X happen?').
- ABDUCTIVE_QUERY: User is asking for likely causes or explanations for an observation ('why did X happen?', 'what could cause Y?'). For this intent, the payload MUST be a simple description of the observed event, suitable for translation into a ProbLog term (e.g., "alarm_rang", "power_outage", "grass_is_wet"). Do NOT include articles like 'the'.
- SHOW_MODEL: User wants to see the current rules/facts.
- HELP: User is asking for help.
- QUIT: User wants to exit.
- UNKNOWN: The intent is unclear or none of the above.

Respond ONLY with a JSON object containing 'intent' and 'payload' (the extracted statement, question, or simple observed event term, or null if not applicable).

Examples:
User: "It is sunny with 70% probability" -> {{"intent": "ADD_FACT", "payload": "It is sunny with 70% probability"}}
User: "If it rains, the grass gets wet." -> {{"intent": "ADD_FACT", "payload": "If it rains, the grass gets wet."}}
User: "What is the chance of rain?" -> {{"intent": "DEDUCTIVE_QUERY", "payload": "What is the chance of rain?"}}
User: "Will the alarm sound?" -> {{"intent": "DEDUCTIVE_QUERY", "payload": "Will the alarm sound?"}}
User: "The alarm is ringing. Why?" -> {{"intent": "ABDUCTIVE_QUERY", "payload": "alarm_ringing"}}  # Note: Simple term-like event
User: "What might cause the power outage?" -> {{"intent": "ABDUCTIVE_QUERY", "payload": "power_outage"}} # Note: Simple term-like event
User: "Show me the rules" -> {{"intent": "SHOW_MODEL", "payload": null}}
User: "help" -> {{"intent": "HELP", "payload": null}}
User: "exit" -> {{"intent": "QUIT", "payload": null}}
User: "Tell me a joke" -> {{"intent": "UNKNOWN", "payload": "Tell me a joke"}}

User request: "{user_input}"
JSON response:"""

        try:
            response = llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are an intent classification assistant. Respond ONLY with the JSON object as described."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.0,
                response_format={"type": "json_object"},
                n=1,
                stop=None,
            )
            result_json_str = response.choices[0].message.content.strip()

            try:
                result_data = json.loads(result_json_str)
                intent = result_data.get("intent", "UNKNOWN").upper()
                payload = result_data.get("payload")
                if intent not in ["ADD_FACT", "DEDUCTIVE_QUERY", "ABDUCTIVE_QUERY", "SHOW_MODEL", "HELP", "QUIT", "UNKNOWN"]:
                    print(f"Warning: LLM returned unexpected intent '{intent}'. Treating as UNKNOWN.")
                    return "UNKNOWN", user_input
                # If payload is None for intents that need it, return original input
                if payload is None and intent in ["ADD_FACT", "DEDUCTIVE_QUERY", "ABDUCTIVE_QUERY"]:
                     print(f"Warning: LLM returned null payload for intent '{intent}'. Using full input.")
                     payload = user_input
                return intent, payload

            except json.JSONDecodeError:
                print(f"Warning: LLM did not return valid JSON for intent recognition. Raw: '{result_json_str}'")
                return "UNKNOWN", user_input
            except Exception as e:
                 print(f"Warning: Error processing LLM intent response: {e}")
                 return "UNKNOWN", user_input

        except OpenAIError as e:
            print(f"Error calling LLM API for intent recognition: {e}")
            return "UNKNOWN", user_input
        except Exception as e:
            print(f"An unexpected error occurred during intent recognition: {e}")
            return "UNKNOWN", user_input

    def process_input(self, user_input: str) -> tuple[str, str]:
        """
        Processes the user's natural language input, determines intent,
        interacts with the ProbLog interface, and returns a user-friendly response.

        Args:
            user_input (str): The raw user input.

        Returns:
            tuple[str, str]: A tuple containing the response type ('response', 'quit', 'error')
                             and the message to display to the user.
        """
        intent, payload = self._get_intent_and_payload(user_input)

        response_message = ""
        response_type = "response" # Default response type

        if intent == "QUIT":
            response_message = "Exiting."
            response_type = "quit"
        elif intent == "HELP":
            # Help text is handled by the CLI part, just signal intent
            response_type = "help"
            response_message = "" # No specific message needed here
        elif intent == "SHOW_MODEL":
            model_str = self.interface.get_model_string()
            if model_str:
                response_message = f"\n--- Current ProbLog Model ---\n{model_str}\n---------------------------\n"
            else:
                response_message = "\n--- Current ProbLog Model ---\n(Model is empty)\n---------------------------\n"
        elif intent == "ADD_FACT":
            if payload:
                # TODO: Implement smarter rule management here (Phase 3 from previous plan)
                # For now, just add directly and report back
                # Capture print output from add_fact_nl to include in response
                # This is a bit hacky, ideally add_fact_nl would return status/message
                import io, contextlib
                f = io.StringIO()
                with contextlib.redirect_stdout(f):
                    self.interface.add_fact_nl(payload)
                add_output = f.getvalue().strip()
                if "LLM translation added" in add_output:
                     response_message = f"Okay, I've added that to the knowledge base: {add_output.split(': ')[1]}"
                elif "Warning: LLM output" in add_output:
                     response_message = f"Sorry, I tried to add that, but the translation didn't look like valid ProbLog: {payload}"
                else:
                     response_message = f"Sorry, I couldn't add that fact/rule: {payload}"

            else:
                response_message = "Sorry, I understood you wanted to add a fact, but couldn't extract the statement."
                response_type = "error"
        elif intent == "DEDUCTIVE_QUERY":
            if payload:
                explanation = self.interface.query_deductive_nl_explained(payload)
                response_message = f"Analysis: {explanation}"
            else:
                response_message = "Sorry, I understood you wanted to ask a 'what if' question, but couldn't extract the question."
                response_type = "error"
        elif intent == "ABDUCTIVE_QUERY":
             # Pass the original user input to the explanation method,
             # as _translate_nl_to_evidence works better with full sentences.
             # The 'payload' from intent recognition might be too processed (e.g., 'alarm_ringing').
             explanation = self.interface.query_abductive_nl_explained(user_input)
             response_message = f"Analysis: {explanation}"
             # We ignore the extracted 'payload' here for abduction.
             # if not payload: # This check is less relevant now
             #    response_message = "Sorry, I understood you wanted to ask a 'why' question, but couldn't extract the core observation."
             #    response_type = "error"
        elif intent == "UNKNOWN":
            response_message = f"Sorry, I wasn't sure how to handle that: '{user_input}'\nYou can state facts, ask 'what if' (probability), ask 'why' (causes), ask to 'show facts', or 'help'."
            response_type = "error"

        return response_type, response_message

# Example usage (optional, for direct testing)
if __name__ == '__main__':
    if not llm_client:
        print("LLM Client not available. Cannot run example.")
    else:
        librarian = ExpertSystemLibrarian(debug=True)
        print("Testing Librarian directly...")

        type1, resp1 = librarian.process_input("It rains 50% of the time.")
        print(f"[{type1}] {resp1}")

        type2, resp2 = librarian.process_input("What is the probability it rains?")
        print(f"[{type2}] {resp2}")

        type3, resp3 = librarian.process_input("Show the facts")
        print(f"[{type3}] {resp3}")

        type4, resp4 = librarian.process_input("quit")
        print(f"[{type4}] {resp4}")
