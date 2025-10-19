import click
import json
from pathlib import Path

from featureinterp.record import SAEIndexRecord, SAEIndexId
from featureinterp import utils


DATASET_PATH = "data/pile-uncopyrighted_gemma-2-9b/records"


def load_record(layer: int, feature: int) -> SAEIndexRecord:
    """Load the record for a specific layer and feature."""
    hook_dir = f"blocks.{layer}.hook_resid_post"
    record_path = Path(DATASET_PATH) / hook_dir / f"{feature}.json"
    
    with open(record_path) as f:
        data = json.load(f)
        
    record = SAEIndexRecord(
        id=SAEIndexId(layer_index=layer, latent_index=feature),
        max_expression=data["max_expression"],
        max_holistic_expression=data["max_holistic_expression"],
        most_act_records=[],  # We'll only load the top records
    )
    
    # Load just the top 10 records
    for record_data in data["most_act_records"][:10]:
        record.most_act_records.append({
            "tokens": record_data["tokens"],
            "expressions": record_data["expressions"],
            "dataset_index": record_data["dataset_index"]
        })
    
    return record


def display_record(record: SAEIndexRecord):
    """Display the top 10 most activating records with their expressions."""
    print(f"\nFeature {record.id.latent_index} in Layer {record.id.layer_index}")
    print("=" * 80)
    
    for i, rec in enumerate(record.most_act_records[:10], 1):
        print(f"\n{i}. Record {rec['dataset_index']}")
        expressions = [float(x) for x in rec['expressions']]  # Convert to float
        max_expr = max(expressions)
        normalized_expr = [x/max_expr for x in expressions]
        print(utils.render_expressions(rec['tokens'], normalized_expr))


@click.command()
@click.argument('layer', type=int, default=10)
@click.argument('feature', type=int, default=6)
def main(layer: int, feature: int):
    """Display the top-10 most activating sentences for a given layer and feature."""
    try:
        record = load_record(layer, feature)
        display_record(record)
    except FileNotFoundError:
        print(f"Error: Could not find record for layer {layer}, feature {feature}")
        print(f"Please check that the path exists: {DATASET_PATH}/blocks.{layer}.hook_resid_post/{feature}.json")


if __name__ == '__main__':
    main()
