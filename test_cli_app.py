import unittest
import subprocess
import sys
import os
import time
import re # Import regex
from llm_interface import client as llm_client # Import client to check availability
from statement_equality_using_llm import verify_conceptual_match # Import the utility function

# Check if LLM client is initialized (requires OPENROUTER_API_KEY)
LLM_CLIENT_AVAILABLE = llm_client is not None

# Define the path to the cli_app.py script
CLI_SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "cli_app.py")

# --- Test Class ---

@unittest.skipUnless(LLM_CLIENT_AVAILABLE, "LLM client not initialized (OPENROUTER_API_KEY not set or invalid)")
class TestCliAgentApp(unittest.TestCase): # Renamed class

    def _run_agent_test(self, inputs: list[str], timeout: int = 90):
        """Helper function to run the agent script with inputs and return output."""
        input_str = "\n".join(inputs) + "\n"

        process = subprocess.Popen(
            [sys.executable, CLI_SCRIPT_PATH],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            bufsize=1
        )

        try:
            stdout_data, stderr_data = process.communicate(input=input_str, timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout_data, stderr_data = process.communicate()
            self.fail(f"CLI agent script timed out. Stderr: {stderr_data}\nStdout: {stdout_data}")

        # Print output for debugging if needed
        # print("\n--- Agent STDOUT ---")
        # print(stdout_data)
        # print("--- Agent STDERR ---")
        # print(stderr_data)
        # print("--------------------\n")


        # Basic error checks
        self.assertNotIn("Error calling LLM API", stderr_data, f"Stderr contains LLM API errors. Stderr: {stderr_data}")
        self.assertNotIn("Traceback", stderr_data, f"Stderr contains tracebacks. Stderr: {stderr_data}")
        # Allow warnings, e.g., about JSON parsing or unexpected intents
        self.assertEqual(process.returncode, 0, f"CLI agent script exited with non-zero code: {process.returncode}. Stderr: {stderr_data}")

        return stdout_data, stderr_data

    def test_agent_add_deductive_show(self):
        """Tests adding facts, deductive query, and showing model via NL."""
        inputs = [
            "There is a 40% chance it is cloudy", # ADD_FACT
            "If it's cloudy, it might rain with 75% probability", # ADD_FACT
            "What is the probability it might rain?", # DEDUCTIVE_QUERY
            "Show me the facts", # SHOW_MODEL
            "quit" # QUIT
        ]
        stdout_data, stderr_data = self._run_agent_test(inputs)

        # Check for intent understanding messages (REMOVED - check final responses instead)
        # self.assertIn("Understood as adding fact/rule...", stdout_data)
        # self.assertIn("Understood as a 'what if' (probability) query...", stdout_data)

        # Check fact addition confirmations (Now part of the librarian's response)
        self.assertTrue(re.search(r"Okay, I've added.*cloudy", stdout_data), "Cloudy fact addition response missing")
        self.assertTrue(re.search(r"Okay, I've added.*rain.*:-\s*cloudy", stdout_data), "Rain rule addition response missing")

        # Check deductive query result using LLM verification
        self.assertIn("Analysis:", stdout_data)
        deductive_concept = "The probability of rain is approximately 30%, derived from the chance of clouds and the rule connecting clouds to rain."
        # Extract the relevant part of the output for verification
        analysis_output = ""
        if "Analysis:" in stdout_data:
             analysis_output = stdout_data.split("Analysis:", 1)[1].split(">")[0].strip() # Get text after Analysis: until next prompt
        self.assertTrue(verify_conceptual_match(analysis_output, deductive_concept),
                        f"Deductive query output did not conceptually match expected concept.\nOutput: {analysis_output}\nConcept: {deductive_concept}")

        # Check show model output (Keep simple checks for structure)
        self.assertIn("--- Current ProbLog Model ---", stdout_data)
        self.assertIn("cloudy.", stdout_data) # Base fact term
        self.assertIn("rain", stdout_data) # Term from the rule
        self.assertIn(":-", stdout_data) # Rule indicator
        self.assertIn("---------------------------", stdout_data)

        # Check exit message
        self.assertIn("Exiting.", stdout_data)

    def test_agent_abductive_query(self):
        """Tests abductive query ('why') via NL."""
        inputs = [
            "Burglary happens 10% of the time", # ADD_FACT
            "Earthquakes happen 5% of the time", # ADD_FACT
            "If a burglary happens, the alarm rings 95% of the time", # ADD_FACT
            "If an earthquake happens, the alarm rings 80% of the time", # ADD_FACT
            "Why did the alarm ring?", # ABDUCTIVE_QUERY
            "exit" # QUIT
        ]
        stdout_data, stderr_data = self._run_agent_test(inputs)

        # Check for intent understanding (REMOVED - check final responses instead)
        # self.assertIn("Understood as a 'why' (causes) query...", stdout_data)

        # Check fact additions
        self.assertTrue(re.search(r"Okay, I've added.*burglary", stdout_data), "Burglary fact addition response missing")
        self.assertTrue(re.search(r"Okay, I've added.*earthquake", stdout_data), "Earthquake fact addition response missing")
        self.assertTrue(re.search(r"Okay, I've added.*alarm :- burglary", stdout_data), "Alarm/Burglary rule addition response missing")
        self.assertTrue(re.search(r"Okay, I've added.*alarm :- earthquake", stdout_data), "Alarm/Earthquake rule addition response missing")


        # Check abductive query result
        self.assertIn("Analysis:", stdout_data)
        # Ensure the error message from the previous run is gone
        self.assertNotIn("No clauses found for", stdout_data)
        self.assertNotIn("couldn't determine the likely causes", stdout_data)

        # Check abductive query result using LLM verification
        abductive_concept = "Given the alarm rang, burglary is the more likely cause (around 70-75%) compared to earthquake (around 30-35%)."
        # Extract the relevant part of the output for verification
        analysis_output_abduction = ""
        if "Analysis:" in stdout_data:
             analysis_output_abduction = stdout_data.split("Analysis:", 1)[1].split(">")[0].strip() # Get text after Analysis: until next prompt
        self.assertTrue(verify_conceptual_match(analysis_output_abduction, abductive_concept),
                        f"Abductive query output did not conceptually match expected concept.\nOutput: {analysis_output_abduction}\nConcept: {abductive_concept}")

        # Check exit message
        self.assertIn("Exiting.", stdout_data)

    def test_agent_help_unknown(self):
        """Tests help and unknown intents."""
        inputs = [
            "help", # HELP
            "Tell me about the weather", # UNKNOWN
            "quit" # QUIT
        ]
        stdout_data, _ = self._run_agent_test(inputs, timeout=30) # Shorter timeout

        # Check help output
        self.assertIn("How I can help:", stdout_data)
        self.assertIn("State facts or rules", stdout_data)
        self.assertIn("Ask 'what if' questions", stdout_data)
        self.assertIn("Ask 'why' questions", stdout_data)

        # Check unknown intent handling
        self.assertIn("Sorry, I wasn't sure how to handle that:", stdout_data)
        self.assertIn("Tell me about the weather", stdout_data)

        # Check exit message
        self.assertIn("Exiting.", stdout_data)

    def test_agent_clue_scenario(self):
        """Tests reasoning over a more complex 'Clue'-style scenario using exact user input."""
        inputs = [
            # Facts exactly as provided by user
            "A murder has taken place at Blackwood Manor.",
            "The victim, Mr. Everett, was found dead in the library at 10:00 PM.",
            "Six people were present at the manor during the time of the murder: Mrs. White (the housekeeper), Professor Plum (a colleague), Colonel Mustard (a family friend), Miss Scarlet (Mr. Everett's niece), Reverend Green (the local vicar), and Dr. Peacock (Mr. Everett's physician).",
            "The murder weapon was determined to be either a candlestick, a wrench, or a knife.",
            "At 9:30 PM, Mrs. White was preparing dinner in the kitchen.",
            "Colonel Mustard and Miss Scarlet were playing chess in the conservatory from 9:15 PM to 9:50 PM.",
            "Professor Plum was seen leaving the library at 9:20 PM, stating he was going to the study to retrieve some research papers.",
            "Dr. Peacock claims to have been in the basement examining the wine collection from 9:25 PM to 9:55 PM.",
            "Reverend Green says he was alone in the dining room from 9:15 PM to 9:45 PM, preparing for dinner.",
            "A wet set of footprints was found leading from the garden entrance to the library.",
            "It had been raining heavily since 8:00 PM.",
            "Mrs. White confirms that no one came into the kitchen between 9:00 PM and 10:00 PM.",
            "A candlestick is missing from the dining room's candelabra.",
            "A bloody wrench was found hidden in the garden bushes.",
            "Colonel Mustard has a cut on his right hand that he claims happened while fixing his car earlier that day.",
            "Miss Scarlet inherited a substantial amount of money upon Mr. Everett's death.",
            "Professor Plum had a heated argument with Mr. Everett earlier that day regarding academic research.",
            "Dr. Peacock was heard telling Mr. Everett that 'this would be his last warning' during afternoon tea.",
            "Reverend Green's muddy shoes were found in the coat closet near the library.",
            "The library has two entrances: one from the main hall and one from the garden.",
            "The medical examiner determined the time of death to be between 9:30 PM and 9:45 PM.",
            "The knife from the kitchen was found clean but had recently been washed.",
            # Query
            "Who was the killer, and with what weapon? What is your reasoning?",
            "quit"
        ]

        # Use a longer timeout due to the large number of facts and complex query
        stdout_data, stderr_data = self._run_agent_test(inputs, timeout=240)

        clue_query_answer = "The analysis should indicate that the murderer is Reverend Green who killed Mr. Everett with the candlestick in the library. The most logical conclusion is that Reverend Green took the candlestick from the dining room, entered the garden in the rain (leaving his shoes muddy), entered the library through the garden entrance (leaving wet footprints), murdered Mr. Everett between 9:30-9:45 PM, and then removed his muddy shoes and placed them in the closet near the library."

        # Check query result using LLM verification
        self.assertIn("Analysis:", stdout_data)
        analysis_output = ""
        if "Analysis:" in stdout_data:
             # Find the last occurrence of "Analysis:" to get the query result
             analysis_sections = stdout_data.split("Analysis:")
             if len(analysis_sections) > 1:
                 analysis_output = analysis_sections[-1].split(">")[0].strip()

        # We expect this test to FAIL initially, as the system likely cannot handle this complexity yet.
        # The verify_conceptual_match function will use an LLM to check the output.
        self.assertTrue(verify_conceptual_match(analysis_output, clue_query_answer),
                        f"'Clue' query output did not conceptually match expected concept.\nOutput: {analysis_output}\nConcept: {clue_query_answer}")

        # Check exit message
        self.assertIn("Exiting.", stdout_data)


if __name__ == '__main__':
    # Ensure the script can find llm_interface relative to its location
    sys.path.insert(0, os.path.dirname(CLI_SCRIPT_PATH))
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
