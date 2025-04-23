# CausalGraphTools

A framework for modeling, analyzing, and visualizing complex causal systems with probabilistic logical inference and AI-enhanced rule generation.

## Overview

CausalGraphTools combines probabilistic logic programming with large language models to enable real-time causal reasoning across domains. The system uses natural language as its primary interface for both inputs and outputs, allowing users to interact with complex causal models through conversation while leveraging powerful logical inference capabilities beneath the surface.

## Core Thesis

Expert systems failed not because of technical limitations, but due to the "knowledge acquisition bottleneck" - the immense human effort required to encode and maintain rule bases. CausalGraphTools solves this by using LLMs to transform unstructured information into structured probabilistic rules within a causal graph, while also providing a natural language interface for all interactions.

## Architecture

### 1. Knowledge Representation Layer (ProbLog/DeepProbLog)

The core of the system where causal relationships are encoded as probabilistic logical statements. This layer:

- Represents nodes (entities) and edges (causal relations) with associated probabilities
- Supports modeling of first, second, and third-order effects
- Enables definition of feedback loops and complex conditional dependencies
- Allows integration of domain-specific rules and constraints

### 2. Inference Engine

Built on ProbLog's reasoning capabilities, the inference engine:

- Performs deductive reasoning to predict outcomes from given inputs
- Uses abductive reasoning to identify likely causes of observed phenomena
- Applies inductive learning to refine model weights from data
- Identifies patterns and structures within the causal graph
- Calculates sensitivity metrics to identify critical intervention points

### 3. LLM Interface Layer

Serves as the unified natural language interface for all interactions:

- Transforms raw observations into structured causal rules
- Translates user queries into formal inference tasks
- Communicates results and insights in natural language
- Explores potential interventions through common-sense reasoning
- Handles all input/output through conversational interaction

### 4. Visualization Layer (Cytoscape.js)

A supplementary visual interface that:

- Renders the causal graph with weighted nodes and edges
- Updates dynamically based on inference results
- Supports exploration of complex causal structures
- Provides an alternative view for understanding system relationships

## Core Use Cases

### 1. Scenario Playthrough (Deductive Reasoning)
Users describe hypothetical conditions and the system predicts likely outcomes by propagating probabilities through the causal graph.

**Example interaction:**
```
User: "What would happen if the EU imposed a 25% tariff on Chinese EVs?"

System: "Based on the causal model, a 25% EU tariff on Chinese EVs would 
likely lead to:
1. Increased EV prices in the EU market (87% probability)
2. Reduced EV adoption rate in Europe (74% probability)
3. Increased European EV production (65% probability)
4. Retaliatory measures from China targeting European luxury goods (71% probability)

The second-order effects include potential shifts in supply chains toward 
Southeast Asia (58% probability) and increased tensions in EU-China diplomatic 
relations (69% probability)."
```

### 2. Revealing Causes (Abductive Reasoning)
Users describe observed phenomena and the system identifies the most likely causes or contributing factors.

**Example interaction:**
```
User: "We're seeing unexpected inflation in housing despite rising interest rates. 
What could be causing this?"

System: "The most likely explanations for persistent housing inflation 
despite rising interest rates are:

1. Supply constraints in housing construction (83% probability)
   - Connected to: ongoing materials shortages and labor issues

2. Investment fund activity in residential real estate (76% probability)
   - Connected to: capital seeking inflation hedges

3. Migration patterns to specific high-demand metro areas (72% probability)
   - Connected to: remote work policies and regional economic growth

Would you like me to explore any of these potential causes in more depth?"
```

### 3. Desired Outcome Targeting (Combined Reasoning)
Users specify desired outcomes and constraints, and the system uses a combination of abduction, deduction, and LLM-generated possibilities to suggest effective interventions.

**Example interaction:**
```
User: "How could we reduce carbon emissions by 30% within 5 years while 
maintaining economic growth above 2%?"

System: "Based on causal analysis, here are three viable pathways to reduce 
carbon emissions by 30% while maintaining 2%+ economic growth:

1. Accelerated renewable energy transition + carbon pricing
   - Key intervention: $60-80/ton carbon price phased in over 3 years
   - Probability of meeting both targets: 64%
   - Critical enabling factor: Renewable infrastructure investment

2. Industrial efficiency transformation + transport electrification
   - Key intervention: Mandate 40% electric vehicles in new sales by year 3
   - Probability of meeting both targets: 58%
   - Critical enabling factor: Grid capacity upgrades

3. Targeted methane reduction + building efficiency standards
   - Key intervention: 80% reduction in industrial methane leakage
   - Probability of meeting both targets: 52%
   - Critical enabling factor: Low-cost methane capture technology

Would you like details on any of these pathways or additional alternatives?"
```

## Example Domain Applications

### Climate Dynamics Modeling

- **Input**: News about record temperatures, studies on feedback mechanisms, policy announcements
- **Causal Analysis**: Modeling greenhouse gas emissions, feedback loops, and climate outcomes
- **Key Outputs**: 
  - Identification of critical tipping points and vulnerable systems
  - Evaluation of intervention strategies and their likely effects
  - Prediction of climate trajectory under different scenarios

**Example query**: "If methane emissions continue rising at current rates, how would that affect the timeline for reaching 1.5°C warming?"

### Macroeconomic Analysis

- **Input**: Central bank announcements, economic indicators, market events, fiscal policy changes
- **Causal Analysis**: Modeling relationships between monetary policy, inflation, employment, and growth
- **Key Outputs**:
  - Identification of causal pathways from policy to economic outcomes
  - Analysis of feedback loops between sectors of the economy
  - Prediction of economic indicators under different policy scenarios

**Example query**: "What would be the likely impact on inflation and unemployment if the Fed raises rates by 75 basis points next month?"

### Tariff Negotiation Wargaming

- **Input**: Announcements of tariffs, diplomatic statements, industry responses, trade data
- **Causal Analysis**: Modeling trade relationships, tariff effects, and strategic responses
- **Key Outputs**:
  - Simulation of multi-party negotiation dynamics
  - Identification of leverage points and vulnerabilities
  - Prediction of likely responses to proposed tariff structures

**Example query**: "If we impose a 10% tariff on agricultural imports from Country X, what are the most likely retaliatory measures and how would they affect our manufacturing sector?"

### Public Health Response

- **Input**: Disease surveillance data, intervention studies, behavioral research, policy implementations
- **Causal Analysis**: Modeling disease spread dynamics, intervention effects, and public response
- **Key Outputs**:
  - Evaluation of intervention strategies and their likely effects
  - Identification of critical control points in disease transmission
  - Prediction of health outcomes under different policy scenarios

**Example query**: "What combination of public health measures would most effectively reduce transmission while minimizing economic impact?"

## Getting Started

### Prerequisites
- Python 3.8+
- ProbLog/DeepProbLog
- Node.js (for visualization)
- OpenAI API key or equivalent


### Basic Usage
The primary interface is conversational, with the LLM serving as both input processor and output generator:

```python
from causal_graph_tools import CausalSystem

# Initialize the system
system = CausalSystem(
    llm_api_key="your_api_key",
    initial_domain="macroeconomics"  # Optional starting domain
)

# Start the interactive interface
system.start_interactive_session()

# Alternatively, process a specific query
response = system.process_query(
    "What would happen to housing prices if interest rates rise by 1%?"
)
print(response)

# Add new empirical observations
system.process_observation(
    "Recent Federal Reserve minutes indicate a potential pause in rate hikes 
    due to concerns about financial stability."
)

# Generate visualization (supplementary to the main interface)
system.generate_visualization("causal_graph.html")
```

## Natural Language Interface

The LLM serves as a unified interface for all system interactions:

### Input Processing
- **Empirical Observations**: Users share news, data, or research findings in natural language
- **Queries**: Users ask questions about causes, effects, or scenarios
- **Instructions**: Users can provide feedback or guidance about the model

### Output Generation
- **Explanations**: Clear descriptions of causal relationships and reasoning
- **Predictions**: Probabilistic forecasts with confidence levels
- **Recommendations**: Suggested interventions for desired outcomes
- **Clarifications**: Requests for additional information when needed

### Interface Examples

**Adding Empirical Observations:**
```
User: "A new study from MIT shows that methane emissions from permafrost 
are 60% higher than previous estimates."

System: "I've analyzed this observation and identified several causal implications:

1. Added rule: Permafrost thawing causes increased methane emissions (probability 0.95)
2. Updated rule: Increased methane emissions causes accelerated warming (probability increased from 0.82 to 0.88)
3. Added new connection: Accelerated warming causes further permafrost thawing (probability 0.77)

This creates a positive feedback loop in the climate system. Is there anything specific about these rules you'd like to modify before I incorporate them?"
```

**Performing Complex Analysis:**
```
User: "What are the most vulnerable points in our supply chain given 
current geopolitical tensions?"

System: "Based on abductive analysis of the causal model, the three most 
vulnerable points in your supply chain are:

1. Semiconductor components from Taiwan (vulnerability score: 8.7/10)
   - Critical dependencies: 14 downstream products
   - Key risk factors: China-Taiwan tensions, limited alternative suppliers

2. Rare earth processing for battery materials (vulnerability score: 7.9/10)
   - Critical dependencies: Electric vehicle production, energy storage
   - Key risk factors: Chinese market concentration, processing capacity

3. Shipping routes through the Suez Canal (vulnerability score: 7.4/10)
   - Critical dependencies: European-Asian logistics timeline
   - Key risk factors: Regional instability, limited alternatives

Would you like detailed mitigation strategies for any of these vulnerabilities?"
```

## Advanced Features

### Pattern Recognition

The system can identify important structural patterns in the causal graph:

```python
# Find feedback loops
loops = model.find_feedback_loops()

# Identify critical nodes
critical_nodes = model.calculate_centrality()

# Find potential intervention points
leverage_points = model.find_leverage_points()
```

### Scenario Analysis

Run comparative simulations to evaluate potential futures:

```python
# Define scenarios
baseline = {"co2_emissions": "current_trend"}
mitigation = {"co2_emissions": "reduced_by_50_percent"}

# Compare outcomes
comparison = model.compare_scenarios(baseline, mitigation, 
                                    target="global_temperature", 
                                    timeframe=50)
```

### Real-time Analysis Pipeline

For time-sensitive applications, set up continuous monitoring:

```python
# Define event handlers
@model.on_critical_node_change
def alert_stakeholders(node, old_value, new_value, affected_outcomes):
    if "market_crash_probability" in affected_outcomes:
        if affected_outcomes["market_crash_probability"] > 0.7:
            send_alert_to_risk_team(affected_outcomes)
```

### Multi-paradigm Reasoning
The system combines:
- **Logical inference** for structured reasoning
- **Probabilistic analysis** for handling uncertainty
- **Neural processing** for pattern recognition and natural language understanding

## Design Philosophy: LLMs for Complexity

A core principle guiding the development of CausalGraphTools is to leverage Large Language Models (LLMs) to manage complexity wherever possible, rather than building intricate, hand-coded logic or parsers. This approach aims to:

- **Reduce the "knowledge acquisition bottleneck"**: By using LLMs to interpret natural language inputs and manage the knowledge base, we minimize the need for users to learn complex syntax or for developers to build sophisticated parsing mechanisms (for example, in the translation between natural language and Problog statements/queries).

This means that in scenarios where complexity arises, our primary strategy is to refine the prompts and instructions given to the LLM, allowing the model to handle the intricacies of natural language and knowledge representation, rather than adding complexity to the Python codebase with elaborate parsing rules or state machines.

## Future Development

Future versions will include:

1. **Automated Monitoring**: Direct integration with news APIs, data sources, and research databases
2. **Rule Validation Framework**: Automated validation of rules against empirical data
3. **Collaborative Modeling**: Multi-user support for team-based model development
4. **Natural Language Querying**: Enhanced capabilities for complex causal questions
5. **Domain-specific Extensions**: Specialized modules for climate, economics, geopolitics, etc.

## Contributing

We welcome contributions to extend CausalGraphTools:

1. **Domain-specific knowledge**: Add specialized causal templates for various fields
2. **Inference algorithms**: Implement additional reasoning mechanisms
3. **UI enhancements**: Improve the interaction experience
4. **LLM integration**: Enhance the natural language capabilities

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- ProbLog and DeepProbLog teams for the probabilistic logic programming foundation
- Cytoscape.js contributors for the visualization framework
- Research in causal inference, systems thinking, and probabilistic programming
