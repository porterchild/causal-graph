import re
from problog.program import PrologString
from problog import get_evaluatable
from problog.logic import Term

def likely_individual_causes(model_string: str, evidence_str: str) -> dict[Term, float] | None:
    """
    Calculates the posterior probability P(Cause | Evidence) for potential individual
    causes (defined as base probabilistic facts like `P::fact.`) given evidence.
    This helps identify the likelihood of each potential cause after observing the evidence.

    Args:
        model_string (str): The ProbLog model as a string.
        evidence_str (str): The ProbLog evidence string (e.g., "evidence(alarm, true).").

    Returns:
        dict[Term, float] | None: Dictionary mapping potential causes to their posterior probability,
                                 or None if calculation fails.
    """
    # Identify potential causes (base probabilistic facts) from the model string
    potential_causes = []
    # Regex to find lines like '0.X::fact.' or 'P::fact.'
    cause_regex = re.compile(r"^\s*(\d+(\.\d*)?|\.\d+)\s*::\s*([a-z_]\w*)\s*\.\s*$")
    for line in model_string.splitlines():
        match = cause_regex.match(line.strip())
        if match:
            cause_name = match.group(3)
            try:
                # Create a Term object for the cause
                dummy_program = PrologString(f"query({cause_name}).")
                parsed_term = None
                for statement in dummy_program:
                    if statement.functor == 'query':
                        parsed_term = statement.args[0]
                        break
                if parsed_term:
                    potential_causes.append(parsed_term)
                else:
                    print(f"Warning: Could not parse potential cause '{cause_name}' into a Term.")
            except Exception as e:
                print(f"Warning: Error parsing potential cause '{cause_name}': {e}")

    if not potential_causes:
        print("Warning: No potential causes (base probabilistic facts like 'P::fact.') found in the model.")
        return None

    # Construct the temporary model for abduction
    abduction_queries = "\n".join([f"query({cause})." for cause in potential_causes])
    abductive_model_string = f"{model_string}\n{evidence_str}\n{abduction_queries}"
    # print(f"\n--- Evaluating Abductive Model ---\n{abductive_model_string}\n----------------------") # Debug

    try:
        pl = PrologString(abductive_model_string)
        # Use standard evaluation: evaluate() returns P(Query | Evidence)
        results = get_evaluatable().create_from(pl).evaluate()
        # print(f"Raw ProbLog Abductive Result: {results}") # Debug

        if isinstance(results, dict):
            # Filter results to include only the potential causes we queried
            posterior_probabilities = {term: prob for term, prob in results.items() if term in potential_causes}
            return posterior_probabilities
        else:
             print(f"Unexpected result type from ProbLog evaluation: {type(results)}. Expected dict.")
             return None

    except Exception as e:
        print(f"Error during abductive inference: {e}")
        return None
