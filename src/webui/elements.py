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

import gradio as gr
import numpy as np

def create_input_components(default_values, property_names, property_ranges):
    initial_property_type = default_values.get('type', 'Material')

    with gr.Blocks() as interface:
        with gr.Row():
            with gr.Column(scale=2):
                instruction = gr.Textbox(
                    label="Instruction",
                    value=default_values.get('instruction', ''),
                    lines=3,
                    placeholder="Enter your instruction here..."
                )
            with gr.Column(scale=1):
                property_type = gr.Radio(
                    ["Drug", "Material"],
                    label="Property Type",
                    value=initial_property_type,
                    interactive=True
                )

        with gr.Row():
            drug_properties = {}
            for prop in ["HIV", "BBBP", "BACE"]:
                default_value = default_values.get('property', {}).get(prop, "NAN")
                drug_properties[prop] = gr.Radio(
                    [0, 1, "NAN"],
                    label=f"{property_names[prop]} ({prop})",
                    value=default_value if default_value in [0, 1, "NAN"] else "NAN",
                    visible=initial_property_type == "Drug",
                    interactive=True
                )

            material_properties = {}
            for prop in ["CO2", "N2", "O2", "FFV", "TC"]:
                min_val = property_ranges[prop]['min']
                max_val = property_ranges[prop]['max']
                default_value = default_values.get('property', {}).get(prop, 0)
                material_properties[prop] = gr.Slider(
                    label=f"{property_names[prop]} (0 for uncondition)",
                    minimum=0,
                    maximum=max_val,
                    value=default_value if default_value != "NAN" else 0,
                    step=0.1,
                    visible=initial_property_type == "Material",
                    interactive=True
                )

        with gr.Row():
            synthetic_properties = {}
            for prop in ["SC", "SA"]:
                min_val = property_ranges[prop]['min']
                max_val = property_ranges[prop]['max']
                default_value = default_values.get('property', {}).get(prop, (min_val + max_val) / 2)
                synthetic_properties[prop] = gr.Slider(
                    label=f"{property_names[prop]} ({prop})",
                    minimum=min_val,
                    maximum=max_val,
                    value=default_value if not np.isnan(default_value) else (min_val + max_val) / 2,
                    step=(max_val - min_val) / 100,
                    interactive=True
                )

        def update_visibility(property_type):
            return (
                [gr.update(visible=(property_type == "Drug")) for _ in drug_properties.values()] +
                [gr.update(visible=(property_type == "Material")) for _ in material_properties.values()]
            )

        property_type.change(
            update_visibility,
            inputs=[property_type],
            outputs=list(drug_properties.values()) + list(material_properties.values())
        )

        return interface, instruction, property_type, drug_properties, material_properties, synthetic_properties