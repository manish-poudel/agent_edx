import os
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import pickle


class CompanyDataFaiss:
    """
    A class to manage company data and perform similarity searches using FAISS and sentence embeddings.
    It creates and saves a FAISS index for company names, allowing fast retrieval of company CIKs based on a search query.
    """

    def __init__(self):
        """
        Initialize the paths to the FAISS index, DataFrame, and JSON file containing company data.
        """
        self.index_file_path = 'assets/faiss_index.index'  # Path to save/load the FAISS index
        self.df_file_path = 'assets/df.pkl'  # Path to save/load the DataFrame containing company data
        self.json_file_path = 'assets/sec_companies.json'  # Path to the JSON file with company details

    def _create_index_and_save(self):
        """
        Create a FAISS index from the company titles, generate embeddings, and save both the index and DataFrame to files.
        :return: The FAISS index and the DataFrame with company data.
        """

        # Load JSON data from the 'assets' folder and create a DataFrame
        with open(self.json_file_path, 'r') as file:
            data = json.load(file)

        # Convert the JSON data into a DataFrame
        df = pd.DataFrame.from_dict(data, orient='index')

        # Generate embeddings for the company titles using a sentence transformer model
        encoder = SentenceTransformer("all-mpnet-base-v2")
        vectors = encoder.encode(df['title'].tolist())
        dim = vectors.shape[1]  # Dimensionality of the embeddings

        # Build a FAISS index using L2 distance and add the vectors
        index = faiss.IndexFlatL2(dim)
        index.add(vectors)

        # Save the FAISS index to a file for later use
        faiss.write_index(index, self.index_file_path)

        # Save the DataFrame as a pickle file for fast loading in future queries
        with open(self.df_file_path, 'wb') as f:
            pickle.dump(df, f)

        return index, df

    def _load_index_and_df(self):
        """
        Load the FAISS index and DataFrame from saved files.
        :return: The loaded FAISS index and DataFrame.
        """

        # Load the FAISS index from the saved file
        index = faiss.read_index(self.index_file_path)

        # Load the DataFrame from the saved pickle file
        with open(self.df_file_path, 'rb') as f:
            df = pickle.load(f)

        return index, df

    def _query_index(self, index, df, search_query):
        """
        Query the FAISS index for the most similar company to the search query.
        :param index: The FAISS index to search.
        :param df: The DataFrame containing company data.
        :param search_query: The search query (company name) to find a match for.
        :return: The company name and corresponding CIK for the closest match.
        """

        # Load the sentence transformer model to encode the search query
        encoder = SentenceTransformer("all-mpnet-base-v2")

        # Encode the search query and reshape it to match FAISS input format
        vec = encoder.encode([search_query])
        svec = np.array(vec).reshape(1, -1)

        # Search for the closest match in the FAISS index (k=1 returns the nearest match)
        distances, indices = index.search(svec, k=1)

        # Get the index of the closest match and retrieve the corresponding row from the DataFrame
        matched_index = indices[0][0]
        matched_row = df.iloc[matched_index]
        cik = matched_row['cik_str']
        company_name = matched_row['title']

        return company_name, cik

    def _main(self, search_query):
        """
        Main method to run the query process. If the index doesn't exist, it creates one; otherwise, it loads and queries it.
        :param search_query: The company name query to search in the FAISS index.
        """
        if not os.path.exists(self.index_file_path):
            print("Index file not found. Creating new index...")
            index, df = self._create_index_and_save()
        else:
            print("Index file found. Loading index...")
            index, df = self._load_index_and_df()

        # Query the index for the given search query
        company_name, cik = self._query_index(index, df, search_query)
        print(f"Company Name: {company_name}")
        print(f"CIK: {cik}")

    def query(self, search_query):
        """
        Public method to query the FAISS index for a company name.
        :param search_query: The search query (company name) to find the closest match.
        :return: The company name and CIK of the closest match.
        """
        index, df = self._load_index_and_df()
        return self._query_index(index, df, search_query)
