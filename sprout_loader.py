from datasets import load_dataset
import pickle

saved_df_filename = 'sprout_df.pkl'

def load_sprout_data():
    """
    Load the SPROUT dataset from Hugging Face and save it locally for faster access.
    """
    try:
        # Try to load the saved dataframe first
        with open(saved_df_filename, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        # If not found, load from Hugging Face and save locally
        print("Loading SPROUT dataset from Hugging Face...")
        ds = load_dataset("CARROT-LLM-Routing/SPROUT")
        
        # Convert to pandas DataFrame
        df = ds['train'].to_pandas()
        
        # Save for future use
        with open(saved_df_filename, 'wb') as f:
            pickle.dump(df, f)
        
        return df 