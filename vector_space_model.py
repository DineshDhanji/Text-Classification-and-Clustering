import pathlib
import re
from typing import Tuple, Set, List, Dict
from nltk.stem import PorterStemmer
import numpy as np
import pickle
import time

from ordered_set import OrderedSet
from math import log10


class VectorSpaceModel:
    def __init__(
        self,
        documents_path: str = None,
        stop_words_file_path: str = None,
        vsm_index_path: str = "vsm_data.pkl",
    ) -> None:
        self.documents_path: str = documents_path
        self.stop_words_file_path: str = stop_words_file_path
        self.stop_words: OrderedSet = OrderedSet()
        self.vsm_index_path = "vsm_data.pkl"
        self.vsm_index: List[Tuple[str, List[float]]] = []
        self.stemmer: PorterStemmer = PorterStemmer()
        self.load_database_time: float = 0
        self.vocabulary: Dict[str, int] = {}
        self.idf = []

    def initiate(self) -> int:
        vsm_index_path = pathlib.Path(self.vsm_index_path)
        if vsm_index_path.exists():
            print(
                "Preprocessed data was found. The system is going to use the provided indexes."
            )
            start_time = time.time()

            # Load stop words
            if not self.stop_words_file_path:
                print("No stop words file path was given to the model.")
                return -1
            try:
                stop_words_file_path = pathlib.Path(self.stop_words_file_path)

                if not stop_words_file_path.exists():
                    print(
                        f"Stop words file path '{self.stop_words_file_path}' does not exist."
                    )
                    return -2

                self.load_stop_words(stop_words_file_path)
            except:
                # Already handled above.
                pass

            self.load_indexes(vsm_index_path)
            self.load_database_time = time.time() - start_time
            print(f"Total entires in the collection\t{len(self.vsm_index)}")
            return 1
        else:
            print(
                "Preprocessed data not found. The system is going to make new indexes."
            )
            if not self.documents_path:
                print("No document path was given to the model.")
                return -1
            if not self.stop_words_file_path:
                print("No stop words file path was given to the model.")
                return -1

            try:
                folder_path = pathlib.Path(self.documents_path)
                stop_words_file_path = pathlib.Path(self.stop_words_file_path)

                if not folder_path.exists():
                    print(f"Document path '{self.documents_path}' does not exist.")
                    return -1
                if not stop_words_file_path.exists():
                    print(
                        f"Stop words file path '{self.stop_words_file_path}' does not exist."
                    )
                    return -2

                self.load_stop_words(stop_words_file_path)
                start_time = time.time()
                dataset = {}
                total_doc = 0
                try:
                    for file_path in folder_path.iterdir():
                        print(f"Parsing\t{file_path}")
                        total_doc += 1
                        dataset.update(self.parse_document(file_path, dataset))
                except Exception as e:
                    print(f"Error occurred during parse_document: {e}")

                self.make_vsm_index(dataset=dataset, n=total_doc)
                # self.save_indexes()
                self.load_database_time = time.time() - start_time
                print("\nSaving all computed indexes on the drive.")
                print(f"Total entires in the collection\t{len(self.vsm_index)}.")
                return 1

            except Exception as e:
                print(f"Error occurred while initiating the model: {e}")
                return -1

    def load_stop_words(self, file_path: pathlib.Path) -> int:
        try:
            with open(file_path, "r") as stop_words_file:
                for stop_word in stop_words_file:
                    self.stop_words.add(stop_word.strip())
            print("Stop words loaded successfully.")
            return 1
        except Exception as e:
            print(f"Error occurred while loading stop words: {e}")
        return -1

    def parse_document(
        self, file_path: pathlib.Path, dataset: Dict[str, Dict[str, int]]
    ) -> Dict[str, Dict[str, int]] | None:
        """
        Parses the document and updates vector space indexes.
        """
        file_name = file_path.stem
        file_contents = file_path.read_text()
        pattern = r"\b(?![a-zA-Z]+://)[A-Za-z]+\b"

        words = re.findall(pattern, file_contents.lower())
        for token in words:
            # Don't include a single character token or stop words
            if len(token) < 2 or token in self.stop_words or len(token) >= 27:
                continue

            stemmed_token = self.stemmer.stem(token)
            try:
                a = dataset[stemmed_token]
                if file_name not in a.keys():
                    dataset[stemmed_token][file_name] = 1
                else:
                    dataset[stemmed_token][file_name] += 1
            except KeyError:
                dataset[stemmed_token] = {file_name: 1}  # (df, tf)
        return dataset

    def save_indexes(
        self,
        vsm_index_path: str = "vsm_data.pkl",
    ) -> int:
        """
        Save vector space index and vocabulary vector to disk.
        """
        with open(vsm_index_path, "wb") as vsm_index_file:
            pickle.dump(self.vocabulary, vsm_index_file)
            pickle.dump(self.vsm_index, vsm_index_file)
        return 1

    def load_indexes(
        self,
        vsm_index_path: str,
    ) -> int:
        """
        Load vector space index and vocabulary vector from disk.
        """
        try:
            with open(vsm_index_path, "rb") as vsm_index_file:
                self.vocabulary = pickle.load(vsm_index_file)
                self.vsm_index = pickle.load(vsm_index_file)

            print("Indexes loaded successfully.")
            return 1
        except Exception as e:
            print(f"Error occurred while loading indexes: {e}")
            return -1

    def make_vsm_index(self, dataset: Dict[str, Dict[str, int]], n: int) -> None:
        self.vocabulary = {word: index for index, word in enumerate(dataset.keys())}

        # Extract unique document names
        doc_vector = list(set().union(*[doc.keys() for doc in dataset.values()]))

        # Initialize the VSM index
        self.vsm_index = []

        # Iterate over each document
        for doc_name in doc_vector:
            temp_vector = [0.0] * len(self.vocabulary)
            for index, (word, word_vector) in enumerate(dataset.items()):
                if len(word_vector) >=3:
                    if doc_name in word_vector:
                        tf = word_vector[doc_name]
                        idf = log10(n / len(word_vector))
                        self.idf.append(idf)
                        temp_vector[index] = tf * idf
                        
            # self.vsm_index.append(tuple([doc_name, temp_vector]))
            normalized_matrix = self.normalize_vectors(np.array([temp_vector]))[0]
            self.vsm_index.append(tuple([doc_name, normalized_matrix]))
        return None

    def normalize_vectors(self, matrix: np.ndarray) -> np.ndarray:
        """
        Normalize Vectors to unit length.

        Args:
            matrix (np.ndarray): Matrix to be normalized.

        Returns:
            np.ndarray: Normalized matrix.
        """
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / norms

    def transform_query(self, user_query: str) -> Tuple[int, List[int | None]]:
        """
        Transform the user query format.
        """
        # Tokenize the user query
        query_tokens = user_query.lower().split()

        if not query_tokens:
            return 0, "Kindly provide a query to proceed"

        stemmed_tokens = []
        # Apply stemming to the query tokens
        for token in query_tokens:
            if token not in self.stop_words:
                stemmed_tokens.append(self.stemmer.stem(token))
        query_vector = []
        for word_item in self.vocabulary.items():
            if word_item[0] in stemmed_tokens:
                query_vector.append(1)
            else:
                query_vector.append(0)

        return (1, query_vector)

    def retrieve_from_vsm(
        self, user_query: str, alpha: float = 0.005
    ) -> Tuple[List[str], float, str]:
        """
        Retrieve documents based on the transformed user query.
        """
        start_time = time.time()

        # Transform the user query
        transform_query_vector_data = self.transform_query(user_query)

        # Check if the query transformation was successful
        if not transform_query_vector_data[0]:
            return ([], -1.0, "Error.")

        # Extract query vector and calculate its magnitude
        query_vector = np.array(transform_query_vector_data[1])
        query_magnitude = np.linalg.norm(query_vector)

        # Check if the query vector is zero
        if query_magnitude == 0:
            return ([], -1.0, "No relevant document found. (´。＿。｀)")

        results = []
        for vector in self.vsm_index:
            # Extract document vector
            doc_vector = np.array(vector[1])

            # Calculate dot product between document vector and query vector
            dot_product = np.dot(doc_vector, query_vector)

            # Calculate magnitudes of document and query vectors
            doc_magnitude = np.linalg.norm(doc_vector)

            # Calculate combined magnitude
            combined_magnitude = doc_magnitude * query_magnitude

            # cosine_similarity = np.cos(dot_product / combined_magnitude)
            # Calculate cosine similarity
            if combined_magnitude == 0.0:
                cosine_similarity = 0.0
            else:
                cosine_similarity = dot_product / combined_magnitude

            results.append((vector[0], cosine_similarity))

        # Sort results by cosine similarity
        results = sorted(results, key=lambda x: x[1], reverse=True)

        print(f"Before Filtering Result\t{results}")
        # Filter results based on threshold alpha
        results = [result for result in results if result[1] >= alpha]
        print(f"After Filtering Result\t{results}")
        if results:
            docs = [doc[0] for doc in results]
            return (docs, time.time() - start_time, "Retrieved Documents:")

        return ([], -1.0, "No relevant document found. (´。＿。｀)")

    def make_y(self, doc_seq: List[int]) -> np.ndarray:
        """
        Map document sequence to labels.

        Args:
            doc_seq (List[int]): List of integers representing document sequences.

        Returns:
            np.ndarray: Numpy array of labels corresponding to the document sequence.

        Label Mapping:
            Explainable Artificial Intelligence:  0
            Heart Failure: 1
            Time Series Forecasting: 2
            Transformer Model: 3
            Feature Selection: 4
        """
        labels = {
            1: "Explainable Artificial Intelligence",
            2: "Explainable Artificial Intelligence",
            3: "Explainable Artificial Intelligence",
            7: "Explainable Artificial Intelligence",
            8: "Heart Failure",
            9: "Heart Failure",
            11: "Heart Failure",
            12: "Time Series Forecasting",
            13: "Time Series Forecasting",
            14: "Time Series Forecasting",
            15: "Time Series Forecasting",
            16: "Time Series Forecasting",
            17: "Transformer Model",
            18: "Transformer Model",
            21: "Transformer Model",
            22: "Feature Selection",
            23: "Feature Selection",
            24: "Feature Selection",
            25: "Feature Selection",
            26: "Feature Selection",
        }
        a = []
        for i in doc_seq:
            a.append(labels[i])
        return np.array(a)
    


vector_space_model = VectorSpaceModel(
    documents_path="ResearchPapers",
    stop_words_file_path="Stopword-List.txt",
)
vector_space_model.initiate()
print(len(vector_space_model.vocabulary))
