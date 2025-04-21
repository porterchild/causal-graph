import unittest
import os
import math
import io
import contextlib
from llm_interface import ProblogLLMInterface, client # Import client to check for API key
from problog.logic import Term # Import Term for MPE result checking

# Check if LLM client is initialized (requires OPENROUTER_API_KEY)
LLM_CLIENT_AVAILABLE = client is not None

@unittest.skipUnless(LLM_CLIENT_AVAILABLE, "LLM client not initialized (OPENROUTER_API_KEY not set or invalid)")
class TestProblogLLMInterface(unittest.TestCase):

    def test_add_fact_nl_translation(self):
        """Tests if NL statements are correctly translated and added using LLM."""
        interface = ProblogLLMInterface()

        # Test probabilistic fact
        interface.add_fact_nl("It is sunny with 70% probability")
        # We can't assert the exact string due to LLM variability, but check if it was added
        self.assertTrue(len(interface.model_string) > 0, "Model string should not be empty after adding a fact")
        # Optional: More robust check would involve parsing the model string and checking for the term

        # Test rule
        interface.add_fact_nl("If it rains, the grass is wet with 80% probability")
        self.assertTrue(len(interface.model_string) > interface.model_string.find("sunny"), "Model string should have grown")

        # Test simple fact (implicit 1.0 probability)
        interface.add_fact_nl("Fact: It is cloudy.")
        self.assertTrue(len(interface.model_string) > interface.model_string.find("wet_grass"), "Model string should have grown further")

        # Test untranslatable statement (should not add anything)
        initial_model = interface.model_string
        interface.add_fact_nl("This statement has no translation rule.")
        # Allow for potential LLM output that doesn't match the validation regex
        # The key is that it shouldn't be added to the model_string if translation fails or is invalid
        self.assertEqual(initial_model, interface.model_string) # Model should be unchanged if translation fails/is invalid

    def test_query_deductive_nl_simple(self):
        """Tests deductive query on a simple model using LLM translation."""
        interface = ProblogLLMInterface()
        interface.add_fact_nl("It is rainy with 60% probability")
        interface.add_fact_nl("If it rains, the grass is wet with 80% probability")

        # Query probability of a base fact
        prob_rainy = interface.query_deductive_nl("What is the probability that it is rainy?")
        self.assertIsNotNone(prob_rainy)
        self.assertAlmostEqual(prob_rainy, 0.6, places=4)

        # Query probability derived from a rule
        # P(wet_grass) = P(wet_grass | rainy) * P(rainy) + P(wet_grass | ~rainy) * P(~rainy)
        # Assuming P(wet_grass | ~rainy) = 0 (not specified)
        # P(wet_grass) = 0.8 * 0.6 + 0 * 0.4 = 0.48
        prob_wet_grass = interface.query_deductive_nl("What is the probability that the grass is wet?")
        self.assertIsNotNone(prob_wet_grass)
        self.assertAlmostEqual(prob_wet_grass, 0.48, places=4)

    def test_query_deductive_nl_with_evidence(self):
        """Tests deductive query when a fact acts as evidence using LLM translation."""
        interface = ProblogLLMInterface()
        interface.add_fact_nl("It is rainy with 60% probability") # Base probability
        interface.add_fact_nl("If it rains, the grass is wet with 80% probability")
        interface.add_fact_nl("Fact: It is rainy.") # Adds 'rainy.' which makes P(rainy)=1.0

        # Query probability of the evidence fact itself
        prob_rainy = interface.query_deductive_nl("What is the probability that it is rainy?")
        self.assertIsNotNone(prob_rainy)
        self.assertAlmostEqual(prob_rainy, 1.0, places=4)

        # Query probability derived from a rule, given the evidence
        # P(wet_grass | rainy) = 0.8
        prob_wet_grass = interface.query_deductive_nl("What is the probability that the grass is wet?")
        self.assertIsNotNone(prob_wet_grass)
        self.assertAlmostEqual(prob_wet_grass, 0.8, places=4)

    def test_query_deductive_nl_multiple_causes(self):
        """Tests query with multiple rules contributing to the result using LLM translation."""
        interface = ProblogLLMInterface()
        interface.add_fact_nl("It is rainy with 60% probability")
        interface.add_fact_nl("It is sprinklers_on with 20% probability") # Assume independent
        interface.add_fact_nl("If it rains, the grass is wet with 80% probability")
        interface.add_fact_nl("If it sprinklers_on, the grass is wet with 90% probability")

        # Expected probability calculated previously: 0.5736
        prob_wet_grass = interface.query_deductive_nl("What is the probability that the grass is wet?")
        self.assertIsNotNone(prob_wet_grass)
        self.assertAlmostEqual(prob_wet_grass, 0.5736, places=4)


    def test_query_undefined_term(self):
        """Tests querying a term not defined in the model using LLM translation."""
        interface = ProblogLLMInterface()
        interface.add_fact_nl("It is rainy with 60% probability")
        prob_sunny = interface.query_deductive_nl("What is the probability that it is sunny?")
        # Should return 0.0 if the term is translatable but not in the model, or None if not translatable
        # With the LLM, it should translate 'sunny' correctly, and since it's not in the model, ProbLog should give 0.0
        self.assertAlmostEqual(prob_sunny, 0.0, places=4)

    def test_untranslatable_query(self):
        """Tests a query that cannot be translated by the LLM."""
        interface = ProblogLLMInterface()
        interface.add_fact_nl("It is rainy with 60% probability")
        # Craft a query that the LLM is unlikely to translate into a single ProbLog term
        result = interface.query_deductive_nl("Tell me a story about the rain.")
        self.assertIsNone(result, "Untranslatable query should return None")

    def test_debug_mode_output(self):
        """Tests if debug messages are printed when debug=True."""
        interface = ProblogLLMInterface(debug=True)
        nl_fact = "Debug test: fact A with 30% probability"
        nl_query = "Debug test: what is probability of fact A?"

        captured_output = io.StringIO()
        with contextlib.redirect_stdout(captured_output):
            interface.add_fact_nl(nl_fact)
            interface.query_deductive_nl(nl_query)

        output_str = captured_output.getvalue()

        # Check for debug output from add_fact_nl
        self.assertIn(f"[DEBUG] NL Statement: '{nl_fact}'", output_str)
        self.assertIn("[DEBUG] ProbLog Code:", output_str) # Check for the label, content varies

        # Check for debug output from query_deductive_nl
        self.assertIn(f"[DEBUG] NL Query: '{nl_query}'", output_str)
        self.assertIn("[DEBUG] ProbLog Term:", output_str) # Check for the label, content varies

    # --- Abduction Tests ---

    def test_translate_nl_to_evidence(self):
        """Tests the internal translation of NL observations to evidence facts."""
        interface = ProblogLLMInterface()
        # Use the internal method directly for focused testing
        evidence_str = interface._translate_nl_to_evidence("We observed that the grass is wet.")
        self.assertIsNotNone(evidence_str)
        self.assertIn("evidence(wet_grass, true).", evidence_str)

        evidence_str_false = interface._translate_nl_to_evidence("The alarm did not sound.")
        self.assertIsNotNone(evidence_str_false)
        self.assertIn("evidence(alarm, false).", evidence_str_false)

        evidence_str_multi = interface._translate_nl_to_evidence("The patient has a fever and a cough.")
        self.assertIsNotNone(evidence_str_multi)
        self.assertIn("evidence(has_fever, true).", evidence_str_multi)
        self.assertIn("evidence(has_cough, true).", evidence_str_multi)

        # Test untranslatable observation
        evidence_str_untranslatable = interface._translate_nl_to_evidence("This is just a random statement.")
        self.assertIsNone(evidence_str_untranslatable)


    def test_query_abductive_nl_simple(self):
        """Tests abductive query (MPE) on a simple model using LLM translation."""
        interface = ProblogLLMInterface()
        # Classic alarm example
        interface.add_fact_nl("There is a burglary with 10% probability") # 0.1::burglary.
        interface.add_fact_nl("There is an earthquake with 5% probability") # 0.05::earthquake.
        interface.add_fact_nl("If there is a burglary, the alarm sounds with 95% probability") # 0.95::alarm :- burglary.
        interface.add_fact_nl("If there is an earthquake, the alarm sounds with 80% probability") # 0.8::alarm :- earthquake.

        # Observe the alarm sounded
        explanation = interface.query_abductive_nl("The alarm sounded.")
        self.assertIsNotNone(explanation)
        # Expected Posterior Probabilities P(Cause | alarm=true)
        # Need to calculate P(alarm) first.
        # P(alarm) = P(alarm|B,E)P(B)P(E) + P(alarm|B,~E)P(B)P(~E) + P(alarm|~B,E)P(~B)P(E) + P(alarm|~B,~E)P(~B)P(~E)
        # P(alarm|B,E) = 1 - (1-0.95)*(1-0.80) = 1 - 0.05*0.20 = 0.99
        # P(alarm|B,~E) = 0.95
        # P(alarm|~B,E) = 0.80
        # P(alarm|~B,~E) = 0 (assuming alarm only caused by B or E)
        # P(B)=0.1, P(E)=0.05, P(~B)=0.9, P(~E)=0.95
        # P(alarm) = (0.99 * 0.1 * 0.05) + (0.95 * 0.1 * 0.95) + (0.80 * 0.9 * 0.05) + (0 * 0.9 * 0.95)
        # P(alarm) = 0.00495 + 0.09025 + 0.036 + 0 = 0.1312
        # P(burglary | alarm) = P(alarm | burglary) * P(burglary) / P(alarm)
        # P(alarm | burglary) = P(alarm|B,E)P(E|B) + P(alarm|B,~E)P(~E|B)
        # Assuming B and E are independent: P(E|B)=P(E)=0.05, P(~E|B)=P(~E)=0.95
        # P(alarm | burglary) = (0.99 * 0.05) + (0.95 * 0.95) = 0.0495 + 0.9025 = 0.952
        # P(burglary | alarm) = 0.952 * 0.1 / 0.1312 = 0.0952 / 0.1312 ~= 0.7256
        # P(earthquake | alarm) = P(alarm | earthquake) * P(earthquake) / P(alarm)
        # P(alarm | earthquake) = P(alarm|B,E)P(B|E) + P(alarm|~B,E)P(~B|E)
        # Assuming B and E are independent: P(B|E)=P(B)=0.1, P(~B|E)=P(~B)=0.9
        # P(alarm | earthquake) = (0.99 * 0.1) + (0.80 * 0.9) = 0.099 + 0.72 = 0.819
        # P(earthquake | alarm) = 0.819 * 0.05 / 0.1312 = 0.04095 / 0.1312 ~= 0.3121
        expected_posteriors = {Term('burglary'): 0.7256, Term('earthquake'): 0.3121}

        # Compare the calculated posterior probabilities
        self.assertEqual(len(explanation), len(expected_posteriors))
        for term, expected_prob in expected_posteriors.items():
            self.assertIn(term, explanation)
            self.assertAlmostEqual(explanation[term], expected_prob, places=3) # Use 3 places due to potential float variations

    def test_query_abductive_nl_no_alarm(self):
        """Tests abductive query (posterior probabilities) when the evidence contradicts common causes."""
        interface = ProblogLLMInterface()
        interface.add_fact_nl("There is a burglary with 10% probability") # 0.1::burglary.
        interface.add_fact_nl("There is an earthquake with 5% probability") # 0.05::earthquake.
        interface.add_fact_nl("If there is a burglary, the alarm sounds with 95% probability") # 0.95::alarm :- burglary.
        interface.add_fact_nl("If there is an earthquake, the alarm sounds with 80% probability") # 0.8::alarm :- earthquake.

        # Observe the alarm did NOT sound
        explanation = interface.query_abductive_nl("The alarm did not sound.")
        self.assertIsNotNone(explanation)
        # Expected Posterior Probabilities P(Cause | alarm=false)
        # P(~alarm) = 1 - P(alarm) = 1 - 0.1312 = 0.8688
        # P(burglary | ~alarm) = P(~alarm | burglary) * P(burglary) / P(~alarm)
        # P(~alarm | burglary) = 1 - P(alarm | burglary) = 1 - 0.952 = 0.048
        # P(burglary | ~alarm) = 0.048 * 0.1 / 0.8688 = 0.0048 / 0.8688 ~= 0.0055
        # P(earthquake | ~alarm) = P(~alarm | earthquake) * P(earthquake) / P(~alarm)
        # P(~alarm | earthquake) = 1 - P(alarm | earthquake) = 1 - 0.819 = 0.181
        # P(earthquake | ~alarm) = 0.181 * 0.05 / 0.8688 = 0.00905 / 0.8688 ~= 0.0104
        expected_posteriors = {Term('burglary'): 0.0055, Term('earthquake'): 0.0104}

        # Compare the calculated posterior probabilities
        self.assertEqual(len(explanation), len(expected_posteriors))
        for term, expected_prob in expected_posteriors.items():
            self.assertIn(term, explanation)
            self.assertAlmostEqual(explanation[term], expected_prob, places=3) # Use 3 places

    def test_query_abductive_nl_untranslatable(self):
        """Tests abductive query (posterior probabilities) with an untranslatable observation."""
        interface = ProblogLLMInterface()
        interface.add_fact_nl("0.1::burglary.")
        interface.add_fact_nl("0.95::alarm :- burglary.")
        result = interface.query_abductive_nl("What is the meaning of life?")
        self.assertIsNone(result, "Untranslatable observation should return None")


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
