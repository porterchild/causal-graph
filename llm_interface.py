import sys
import os
import re
from openai import OpenAI, OpenAIError
from problog.program import PrologString
from problog import get_evaluatable
from problog.logic import Term, Constant # Add Constant
from problog.errors import ProbLogError
from dotenv import load_dotenv # Import load_dotenv
from problog_extensions import likely_individual_causes # Import the renamed function

# Load environment variables from .env file
load_dotenv()

# Initialize LLM client for OpenRouter
# Reads OPENROUTER_API_KEY from environment (loaded from .env)
try:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    # Verify client can make a simple call (optional, but good practice)
    # try:
    #     client.models.list()
    #     print("LLM client initialized successfully for OpenRouter.")
    # except OpenAIError as e:
    #     print(f"Error verifying LLM client with OpenRouter: {e}")
    #     print("Please check your OPENROUTER_API_KEY and OpenRouter service status.")
    #     client = None # Set client to None if verification fails

except OpenAIError as e:
    print(f"Error initializing LLM client: {e}")
    print("Please ensure the OPENROUTER_API_KEY environment variable is set and valid.")
    client = None # Allow script to load but fail on API calls

class ProblogLLMInterface:
    """
    Manages interaction with a ProbLog model using LLM-based natural language translation via OpenRouter.

    Attributes:
        model_string (str): The current ProbLog model as a string.
        llm_model (str): The LLM model to use for translations (via OpenRouter).
        debug (bool): If True, enables printing of NL input and ProbLog output during translation.
    """
    def __init__(self, initial_model_string="", llm_model="google/gemini-2.5-flash-preview", debug=False): # Add debug flag
        """
        Initializes the interface with an optional initial ProbLog model.

        Args:
            initial_model_string (str): A string containing the initial ProbLog rules.
            llm_model (str): The LLM model identifier to use (via OpenRouter).
            debug (bool): Enable debug printing. Defaults to False.
        """
        self.model_string = initial_model_string
        self.llm_model = llm_model
        self.debug = debug # Store debug flag
        if client is None:
            print("Warning: LLM client not initialized. LLM features will not work.")

    def _get_llm_translation(self, prompt: str, max_tokens=50) -> str | None:
        """Helper function to call the LLM API via OpenRouter."""
        if not client:
            print("Error: LLM client not available.")
            return None
        try:
            response = client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are an expert translator converting natural language to ProbLog syntax. Output ONLY the ProbLog code, without explanations or markdown formatting."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.0, # Deterministic output
                n=1,
                stop=None,
            )
            translation = response.choices[0].message.content.strip()
            # Basic cleanup: remove potential markdown backticks
            translation = re.sub(r"^`+|`+$", "", translation).strip()
            # Remove potential "problog" language specifier if present
            translation = re.sub(r"^problog\s*", "", translation, flags=re.IGNORECASE).strip()
            return translation
        except OpenAIError as e:
            print(f"Error calling LLM API via OpenRouter: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during LLM translation: {e}")
            return None

    def _translate_nl_to_problog(self, nl_statement: str) -> str | None:
        """
        Translates a natural language statement into ProbLog facts/rules using an LLM.

        Args:
            nl_statement (str): The natural language statement.

        Returns:
            str | None: The translated ProbLog fact(s)/rule(s), or None on failure.
        """
        prompt = f"""Translate the following natural language statement into a valid ProbLog fact or rule.
Output ONLY the ProbLog code. Do not include any explanations or markdown formatting.

Examples:
Natural Language: "It is rainy with 60% probability"
ProbLog: 0.6::rainy.

Natural Language: "If it rains, the grass is wet with 80% probability"
ProbLog: 0.8::wet_grass :- rainy.

Natural Language: "Fact: It is cloudy."
ProbLog: cloudy.

Natural Language: "If the alarm sounds and there is a burglary, then the police are called."
ProbLog: police_called :- alarm, burglary.

Statement: "{nl_statement}"
ProbLog:"""
        return self._get_llm_translation(prompt)


    def add_fact_nl(self, nl_statement: str):
        """
        Adds a fact/rule to the model based on a natural language statement, using LLM translation.

        Args:
            nl_statement (str): The natural language statement describing the fact/rule.
        """
        problog_code = self._translate_nl_to_problog(nl_statement)
        if self.debug:
            print(f"[DEBUG] NL Statement: '{nl_statement}'")
            print(f"[DEBUG] ProbLog Code: '{problog_code}'") # Print even if invalid/None for debugging

        if problog_code and problog_code.strip().endswith('.'):
            # Simplified validation: just check if the output is non-empty and ends with a period.
            # This is a very basic check to allow LLM output to be added.
            self.model_string += f"\n{problog_code.strip()}" # Add stripped code to avoid leading/trailing whitespace issues
            print(f"LLM translation added: '{problog_code.strip()}' from '{nl_statement}'")
        else:
             print(f"Warning: LLM output '{problog_code}' doesn't look like valid ProbLog (missing trailing period or empty). Not adding.")
             print(f"Could not add fact from: '{nl_statement}' (LLM translation failed or invalid format)")


    def _translate_nl_query_to_term(self, nl_query: str) -> Term | None:
        """
        Translates a natural language query into a ProbLog Term using an LLM.

        Args:
            nl_query (str): The natural language query.

        Returns:
            Term | None: The translated ProbLog Term, or None on failure.
        """
        prompt = f"""Extract the core ProbLog query term from the following natural language question.
The term should represent the event whose probability is being asked about.
Output ONLY the ProbLog term. Do not include any explanations or markdown formatting.

Examples:
Natural Language: "What is the probability that it is sunny?"
ProbLog Term: sunny

Natural Language: "What is the probability that the grass is wet?"
ProbLog Term: wet_grass

Natural Language: "Is the alarm sounding?"
ProbLog Term: alarm

Question: "{nl_query}"
ProbLog Term:"""
        term_str = self._get_llm_translation(prompt, max_tokens=20)

        if term_str:
            # Basic validation: check if it looks like a valid term name
            if re.match(r"^[a-z_]\w*(\(.*\))?$", term_str):
                try:
                    # Use ProbLog parser to create the Term object robustly
                    # We need a dummy program context to parse a term string
                    dummy_program = PrologString(f"query({term_str}).")
                    # Find the query term within the parsed structure
                    parsed_term = None
                    for statement in dummy_program:
                        if statement.functor == 'query':
                            parsed_term = statement.args[0]
                            break
                    if parsed_term:
                         print(f"LLM translation for query: '{nl_query}' -> Term('{parsed_term}')")
                         return parsed_term
                    else:
                         print(f"Warning: Could not parse LLM output '{term_str}' as a ProbLog term.")
                         return None
                except Exception as e:
                    print(f"Warning: Error parsing LLM output '{term_str}' into Term: {e}")
                    return None
            else:
                print(f"Warning: LLM output '{term_str}' doesn't look like a valid ProbLog term.")
                return None
        else:
            print(f"Could not translate query: '{nl_query}' (LLM translation failed)")
            return None

    def _translate_nl_to_evidence(self, nl_observation: str) -> str | None:
        """
        Translates a natural language observation into ProbLog evidence facts using an LLM.

        Args:
            nl_observation (str): The natural language statement describing the observed evidence.

        Returns:
            str | None: The translated ProbLog evidence fact(s) (e.g., "evidence(fact, true)."),
                      or None on failure. Multiple facts may be separated by newlines.
        """
        prompt = f"""Extract the observed evidence from the following natural language statement and format it as ProbLog evidence facts.
Each fact should be on a new line, ending with a period. Use 'true' for observed events and 'false' for events known not to have occurred.
Output ONLY the ProbLog code. Do not include any explanations or markdown formatting.

Examples:
Natural Language: "We observed that the grass is wet."
ProbLog Evidence: evidence(wet_grass, true).

Natural Language: "Given that the alarm did not sound, what happened?"
ProbLog Evidence: evidence(alarm, false).

Natural Language: "The patient has a fever and a cough."
ProbLog Evidence:
evidence(has_fever, true).
evidence(has_cough, true).

Natural Language: "It's confirmed that there was no burglary."
ProbLog Evidence: evidence(burglary, false).

Statement: "{nl_observation}"
ProbLog Evidence:"""
        # Allow potentially more tokens for multiple evidence facts
        evidence_str = self._get_llm_translation(prompt, max_tokens=100)

        if evidence_str:
            # Basic validation: check if lines look like evidence facts
            lines = evidence_str.strip().split('\n')
            valid_lines = []
            all_valid = True
            # Updated regex to require the comma
            evidence_regex = re.compile(r"^evidence\([a-z_]\w*\s*,\s*(true|false)\)\.$")
            for line in lines:
                line = line.strip()
                if evidence_regex.match(line):
                    valid_lines.append(line)
                elif line: # If line is not empty but doesn't match, it's invalid
                    all_valid = False
                    print(f"Warning: LLM output line '{line}' doesn't look like a valid ProbLog evidence fact.")
                    break # Stop processing if one line is invalid

            if all_valid and valid_lines:
                validated_evidence = "\n".join(valid_lines)
                print(f"LLM translation for evidence: '{nl_observation}' ->\n{validated_evidence}")
                return validated_evidence
            else:
                 print(f"Warning: LLM output for evidence '{evidence_str}' did not contain valid evidence facts.")
                 return None
        else:
            print(f"Could not translate observation to evidence: '{nl_observation}' (LLM translation failed)")
            return None


    def query_deductive_nl(self, nl_query: str) -> float | None:
        """
        Performs deductive inference based on a natural language query, using LLM translation.

        Args:
            nl_query (str): The natural language query.

        Returns:
            float | None: The calculated probability, or None if the query fails or cannot be translated.
        """
        query_term = self._translate_nl_query_to_term(nl_query)
        if self.debug:
            print(f"[DEBUG] NL Query: '{nl_query}'")
            print(f"[DEBUG] ProbLog Term: '{query_term}'") # Print the Term object's string representation

        if not query_term:
            return None

        # Add the query to the model string temporarily
        temp_model_string = self.model_string + f"\nquery({query_term})."
        # print(f"\n--- Evaluating Model ---\n{temp_model_string}\n----------------------") # Debug

        try:
            pl = PrologString(temp_model_string)
            # Use the default evaluation method (compiles to formula, then evaluates)
            # This typically returns a dictionary {Term: probability}
            results = get_evaluatable().create_from(pl).evaluate()
            # print(f"Raw ProbLog Result: {results}") # Debug

            if isinstance(results, dict):
                # Find the probability for our specific query term
                probability = results.get(query_term)
                if probability is None:
                     # Check if the term exists but has 0 probability (e.g., impossible event)
                     # Or if the term wasn't grounded/relevant
                     print(f"Warning: Query term '{query_term}' not found in evaluation results dict, returning None.")
                     # Check if the term exists in the knowledge base at all
                     try:
                         kb = get_evaluatable().create_from(pl)
                         kb.get_node_by_name(query_term) # Throws KeyError if not found
                         # If found but not in results, probability is likely 0.0
                         print(f"Term '{query_term}' exists but has 0 probability.")
                         return 0.0
                     except KeyError:
                         print(f"Term '{query_term}' does not exist in the grounded program.")
                         return None
                     except Exception as check_e:
                         print(f"Error checking term existence: {check_e}")
                         return None

                return probability
            else:
                 # Should not happen with default evaluate() but handle defensively
                 print(f"Unexpected result type from ProbLog evaluation: {type(results)}. Expected dict.")
                 return None

        except ProbLogError as e:
             # Check if the error is specifically about missing clauses for the query term
             if "No clauses found for" in str(e) and str(query_term) in str(e):
                 print(f"Term '{query_term}' is undefined (no clauses found). Returning 0.0 probability.")
                 return 0.0
             else:
                 # Handle other ProbLog errors
                 print(f"ProbLog Error during deductive inference: {e}")
                 # print(f"Model causing error:\n{temp_model_string}") # Debugging
                 return None
        except Exception as e:
            print(f"Unexpected Error during ProbLog deductive inference: {e}")
            # print(f"Model causing error:\n{temp_model_string}") # Debugging
    def query_abductive_nl(self, nl_observation: str) -> dict[Term, float] | None:
        """
        Performs abductive inference based on a natural language observation.
        Calculates the posterior probability P(Cause | Observation) for potential
        causes (defined as base probabilistic facts like `P::fact.`).

        Args:
            nl_observation (str): The natural language statement describing the observed evidence.

        Returns:
            dict[Term, float] | None: A dictionary mapping potential causes (probabilistic facts
                                     in the base model) to their posterior probability given the
                                     observation, or None if abduction fails or cannot be translated.
                                     Example: {Term('burglary'): 0.78, Term('earthquake'): 0.15}
        """
        evidence_str = self._translate_nl_to_evidence(nl_observation)
        if self.debug:
            print(f"[DEBUG] NL Observation: '{nl_observation}'")
            print(f"[DEBUG] ProbLog Evidence: '{evidence_str}'") # Print even if None

        if not evidence_str:
            return None

        # Call the imported function to perform the calculation
        posterior_probabilities = likely_individual_causes(self.model_string, evidence_str)

        if posterior_probabilities is not None:
             # The function already prints the result, no need to print again here
             # print(f"Posterior probabilities for causes given '{nl_observation}': {posterior_probabilities}")
             pass
        # Return the result (which could be None if the helper failed)
        return posterior_probabilities


# Example Usage (for testing purposes - requires OPENROUTER_API_KEY in .env)
if __name__ == "__main__":
    if not client:
        print("\nSkipping example usage as LLM client is not initialized.")
    else:
        print("\n--- Running Example Usage ---")
        interface = ProblogLLMInterface()
        interface.add_fact_nl("It is rainy with 60% probability")
        interface.add_fact_nl("If it rains, the grass is wet with 80% probability")
        interface.add_fact_nl("Fact: It is cloudy.") # Adds 'cloudy.'

        prob_rainy = interface.query_deductive_nl("What is the probability that it is rainy?")
        print(f"Query 'What is the probability that it is rainy?': {prob_rainy}")

        prob_wet_grass = interface.query_deductive_nl("What is the probability that the grass is wet?")
        print(f"Query 'What is the probability that the grass is wet?': {prob_wet_grass}")

        prob_cloudy = interface.query_deductive_nl("What is the probability that it is cloudy?")
        print(f"Query 'What is the probability that it is cloudy?': {prob_cloudy}")

        prob_sunny = interface.query_deductive_nl("What is the probability that it is sunny?")
        print(f"Query 'What is the probability that it is sunny?': {prob_sunny}") # Expected: 0.0

        # --- Deductive with Evidence Example ---
        print("\n--- Deductive with Evidence ---")
        interface_with_evidence = ProblogLLMInterface()
        interface_with_evidence.add_fact_nl("It is rainy with 60% probability")
        interface_with_evidence.add_fact_nl("If it rains, the grass is wet with 80% probability")
        interface_with_evidence.add_fact_nl("Fact: It is rainy.") # Adds 'rainy.'

        prob_wet_grass_given_rain = interface_with_evidence.query_deductive_nl("What is the probability that the grass is wet?")
        print(f"Given 'Fact: It is rainy.', Query 'What is the probability that the grass is wet?': {prob_wet_grass_given_rain}")

        # --- Abductive Example ---
        print("\n--- Abductive Example ---")
        abduction_interface = ProblogLLMInterface()
        abduction_interface.add_fact_nl("There is a burglary with 60% probability") # 0.6::burglary.
        abduction_interface.add_fact_nl("There is an earthquake with 30% probability") # 0.3::earthquake.
        abduction_interface.add_fact_nl("If there is a burglary, the alarm sounds with 90% probability") # 0.9::alarm :- burglary.
        abduction_interface.add_fact_nl("If there is an earthquake, the alarm sounds with 70% probability") # 0.7::alarm :- earthquake.

        # Observe that the alarm sounded
        explanation = abduction_interface.query_abductive_nl("The alarm sounded.")
        print(f"Observation: 'The alarm sounded.' -> MPE: {explanation}")
        # Expected MPE based on calculation: {Term('burglary'): True, Term('earthquake'): False}

        # Observe that the alarm did NOT sound
        explanation_no_alarm = abduction_interface.query_abductive_nl("The alarm did not sound.")
        print(f"Observation: 'The alarm did not sound.' -> MPE: {explanation_no_alarm}")
        # Expected MPE: {Term('burglary'): False, Term('earthquake'): False}

        print("\n--- Example Usage Complete ---")
