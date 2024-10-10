# Copyright 2024 Llamole Team
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import yaml
import numpy as np
import gradio as gr
import random
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem

from src.webui.workflow import load_model_and_tokenizer, process_input, generate
from src.webui.elements import create_input_components

# Load candidates
with open('data/molqa_material_examples.json', 'r') as f:
    material_examples = json.load(f)

with open('data/molqa_drug_examples.json', 'r') as f:
    drug_examples = json.load(f)

# Add type to each example
for example in material_examples:
    example['type'] = 'Material'

for example in drug_examples:
    example['type'] = 'Drug'

# Function to process property values
def process_property(value):
    return 1e-8 if value == 0 else value

# Add type to each example and process property values
for example in material_examples:
    example['type'] = 'Material'
    for prop in ['CO2', 'N2', 'O2', 'FFV']:
        if prop in example['property']:
            example['property'][prop] = process_property(example['property'][prop])

# Combine examples
all_examples = material_examples + drug_examples

# Get default values from the first material example
default_values = drug_examples[0]

# Load property ranges and arguments
with open('data/property_ranges.json', 'r') as f:
    property_ranges = json.load(f)

# with open('config/generate/qwen_material.yaml', 'r') as file:
with open('config/generate/llama_material.yaml', 'r') as file:
    args_dict = yaml.safe_load(file)

# Load model and tokenizer outside the function
model, tokenizer, generating_args = load_model_and_tokenizer(args_dict)

def format_example(example):
    formatted = [example['instruction']]
    
    # Determine if it's a drug or material example based on properties
    is_drug = any(prop in example.get('property', {}) for prop in ["HIV", "BBBP", "BACE"])
    formatted.append("Drug" if is_drug else "Material")
    
    # Handle drug properties
    for prop in ["HIV", "BBBP", "BACE"]:
        value = example.get('property', {}).get(prop, float('nan'))
        formatted.append(value if not np.isnan(value) else "NAN")
    
    # Handle material properties
    for prop in ["CO2", "N2", "O2", "FFV", "TC"]:
        value = example.get('property', {}).get(prop, float('nan'))
        formatted.append(value if not np.isnan(value) else 0)  # 0 represents NAN for material properties
    
    # Handle synthetic properties
    for prop in ["SC", "SA"]:
        value = example.get('property', {}).get(prop, float('nan'))
        formatted.append(value if not np.isnan(value) else float('nan'))
    
    return formatted

# Prepare examples
formatted_examples = [format_example(example) for example in all_examples]

def random_example(examples):
    example = random.choice(examples)
    property_type = example['type']
    
    outputs = [example['instruction'], property_type]
    
    for prop in ["HIV", "BBBP", "BACE"]:
        outputs.append(example['property'].get(prop, "NAN"))
    
    for prop in ["CO2", "N2", "O2", "FFV", "TC"]:
        outputs.append(example['property'].get(prop, 0))
    
    for prop in ["SC", "SA"]:
        outputs.append(example['property'].get(prop, float('nan')))
    
    return outputs

def generate_and_visualize(instruction, property_type, HIV, BBBP, BACE, CO2, N2, O2, FFV, TC, SC, SA):
    properties = {
        "HIV": float('nan') if HIV == "NAN" else HIV,
        "BBBP": float('nan') if BBBP == "NAN" else BBBP,
        "BACE": float('nan') if BACE == "NAN" else BACE,
        "CO2": float('nan') if CO2 == 0 else CO2,
        "N2": float('nan') if N2 == 0 else N2,
        "O2": float('nan') if O2 == 0 else O2,
        "FFV": float('nan') if FFV == 0 else FFV,
        "TC": float('nan') if TC == 0 else TC,
        "SC": SC,
        "SA": SA
    }
    
    # Filter out NaN values
    properties = {k: v for k, v in properties.items() if not np.isnan(v)}
    
    print('instruction', instruction)
    print('properties', properties)
    results = run_molqa(instruction, **properties)
        
    llm_response = results.get('llm_response', 'No response generated')
    llm_smiles = results.get('llm_smiles')
    llm_reactions = results['llm_reactions']
    
    molecule_img = visualize_molecule(llm_smiles) if llm_smiles else None
    
    reaction_steps = []
    reaction_imgs = []
    if llm_reactions:
        for i, reaction_dict in enumerate(llm_reactions):
            reaction = reaction_dict.get('reaction')
            if reaction:
                reaction_steps.append(f"Step {i+1}: {reaction}")
                reaction_imgs.append(visualize_reaction(reaction))
    
    return (
        llm_response,
        llm_smiles if llm_smiles else "No SMILES generated",
        molecule_img,
        gr.JSON(value=reaction_steps, visible=bool(reaction_steps)),
        gr.Gallery(value=reaction_imgs, visible=bool(reaction_imgs))
    )

def run_molqa(instruction: str, **properties) -> dict:
    # Filter out properties with NaN values
    filtered_properties = {k: v for k, v in properties.items() if not np.isnan(v)}
    
    input_data = {
        "instruction": instruction,
        "input": "",
        "property": filtered_properties
    }
    
    dataloader, gen_kwargs = process_input(input_data, model, tokenizer, generating_args)
    generated_results = generate(model, dataloader, gen_kwargs)

    return generated_results

def visualize_molecule(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        img = Draw.MolToImage(mol)
        return np.array(img)
    return np.zeros((300, 300, 3), dtype=np.uint8)

def visualize_reaction(reaction: str) -> np.ndarray:
    rxn = AllChem.ReactionFromSmarts(reaction, useSmiles=True)
    if rxn is not None:
        img = Draw.ReactionToImage(rxn)
        return np.array(img)
    return np.zeros((300, 300, 3), dtype=np.uint8)

# Define property names and their full descriptions
property_names = {
    "HIV": "HIV virus replication inhibition",
    "BBBP": "Blood-brain barrier permeability",
    "BACE": "Human β-secretase 1 inhibition",
    "CO2": "CO2 Perm",
    "N2": "N2 Perm",
    "O2": "O2 Perm",
    "FFV": "Fractional free volume",
    "TC": "Thermal conductivity",
    "SC": "Heuristic Synthetic Scores (SCScore)",
    "SA": "Synthetic Synthetic Scores (SAScore)"
}

# Define outputs
outputs = [
    gr.Textbox(label="Overall LLM Response"),
    gr.Textbox(label="Generated SMILES"),
    gr.Image(label="Generated Molecule"),
    gr.JSON(label="Reaction Steps"),
    gr.Gallery(label="Reaction Visualizations")
]

with gr.Blocks() as iface:
    gr.Markdown("# Llamole Demo Interface")
    gr.Markdown("Enter an instruction and property values to generate a molecule design.")

    interface, instruction, property_type, drug_properties, material_properties, synthetic_properties = create_input_components(default_values, property_names, property_ranges)

    random_btn = gr.Button("Random Example")
    generate_btn = gr.Button("Generate")
    
    for output in outputs:
        output.render()
    
    # Update the inputs for the generate button
    all_inputs = [instruction, property_type]
    all_inputs.extend(drug_properties.values())
    all_inputs.extend(material_properties.values())
    all_inputs.extend(synthetic_properties.values())

    generate_btn.click(generate_and_visualize, inputs=all_inputs, outputs=outputs)
    random_btn.click(
        random_example,
        inputs=gr.State(all_examples),
        outputs=all_inputs
    )

if __name__ == "__main__":
    iface.launch(share=True)