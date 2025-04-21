import sys
from problog.program import PrologString
from problog import get_evaluatable
from problog.logic import Term

# Define the original ProbLog model
model_string = r"""
0.6::rainy.
0.3::cloudy.

0.8::wet_grass :- rainy.
0.1::wet_grass :- cloudy.
0.9::go_outside :- \+rainy.

evidence(cloudy).
query(wet_grass).
query(go_outside).
"""

# --- Deductive Inference ---
print("--- Deductive Inference ---")
try:
    # Parse the ProbLog model
    pl = PrologString(model_string)

    # Get the evaluatable knowledge base
    knowledge = get_evaluatable().create_from(pl)

    # Define queries explicitly
    queries_to_evaluate = [Term('wet_grass'), Term('go_outside')]
    results = {}

    print("Probabilities given evidence(cloudy):")
    for query_term in queries_to_evaluate:
        try:
            # Get the internal node index for the query
            query_node_index = knowledge.get_node_by_name(query_term)
            # Evaluate the query node index
            result_value = knowledge.evaluate(query_node_index)
            results[query_term] = result_value
            print(f"  P({query_term}) = {result_value:.4f}")
        except KeyError:
            print(f"  Query '{query_term}' not found in the ground program.")
        except Exception as eval_e:
             print(f"  Error evaluating query '{query_term}': {eval_e}")

except Exception as e:
    print(f"Error during deductive inference setup: {e}")
    sys.exit(1)


# --- Abductive Inference using problog_extensions ---
print("\n--- Abductive Inference (using problog_extensions) ---")
# Define the base model without evidence or cause queries
base_model_string = r"""
0.6::rainy.
0.3::cloudy.

0.8::wet_grass :- rainy.
0.1::wet_grass :- cloudy.
0.9::go_outside :- \+rainy.
"""
# Define the evidence
evidence_string = "evidence(wet_grass, true)."

try:
    # Import the function from the extensions file
    from problog_extensions import likely_individual_causes

    # Call the function
    posterior_probs = likely_individual_causes(base_model_string, evidence_string)

    if posterior_probs:
        print("Posterior probabilities P(Cause | wet_grass):")
        causes = []
        for cause_term, prob in posterior_probs.items():
            print(f"  P({cause_term}|wet_grass) = {prob:.4f}")
            causes.append((str(cause_term), prob))

        # Find and print the most probable explanation from the results
        if causes:
            mpe_cause, mpe_prob = max(causes, key=lambda item: item[1])
            print(f"\nMost probable explanation: {mpe_cause} with P({mpe_cause}|wet_grass) = {mpe_prob:.4f}")
        else:
            print("\nNo causes found in the posterior probabilities.")
    else:
        print("Abductive inference failed.")

except ImportError:
    print("Error: Could not import 'likely_individual_causes' from 'problog_extensions.py'.")
    print("Please ensure the file exists and is in the correct path.")
    sys.exit(1)
except Exception as e:
    print(f"Error during abductive inference using extension: {e}")
    sys.exit(1)

# Alternative approach using explanation.py - Experimental
print("\n--- Alternative Abductive Inference using ProbLog Explanations ---")
try:
    from problog.logic import Term
    from problog.program import PrologString
    from problog.engine import DefaultEngine
    from problog.formula import LogicFormula
    
    # Setup the model
    model = PrologString("""
    0.6::rainy.
    0.3::cloudy.
    
    0.8::wet_grass :- rainy.
    0.1::wet_grass :- cloudy.
    """)
    
    # Create a logic formula
    engine = DefaultEngine()
    db = engine.prepare(model)
    
    # Query for explanations 
    target = Term('wet_grass')
    
    # Generate explanations using engine.explanations
    explanations = engine.explanations(db, target)
    
    print(f"Explanations for {target}:")
    for i, explanation in enumerate(explanations, 1):
        # Convert each explanation to a readable format
        explanation_str = ", ".join([str(lit) for lit in explanation])
        print(f"  Explanation {i}: {explanation_str}")
    
    # Note: For full MPE, you would compute the probability of each explanation
    # and select the most probable one

except Exception as e:
    print(f"Error during alternative abductive inference: {e}")
    print("Note: This functionality may not be available in your ProbLog version.")

print("\nTest completed successfully.")
