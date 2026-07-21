from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import awesome_cossim_topn
import re


def ngrams(string, n=2):
    """
    Custom analyzer for tfidf vectorizer. Splits a
    string into all possible ngrams.

    Parameters
    ---------
        string : str
            String for which we wish to obtain the
            ngrams.
        n : int
            Number of grams.

    Returns
    ---------
        ngrams : list[str]
            List of ngrams for the string.
    """
    # Clean the string.
    string = re.sub(r"[,-/]|\sBD", r"", string)
    # Get the character ngrams.
    ngrams = zip(*[string[i:] for i in range(n)])

    return ["".join(ngram) for ngram in ngrams]


class Keywords:
    """
    Object for storing and matching
    to a predefined keyword list.

    Attributes
    ---------
        words : list
            Words against which we
            want to match.
        ids : list (optional)
            List of meaningful
            ids for alternative
            retrieval of matched words.

    Methods
    ---------
        match
            Match to a single
            word.
    """

    def __init__(self, words: list, ids: list = None):
        """
        Initialize ie train the keyword
        matcher.
        """
        # Tfidf vectorizer.
        self.vectorizer = None
        # Tfidf matrix.
        self.matrix = None
        # Wordlist.
        self.words = words
        # Ids if provided.
        if ids:
            self.ids = ids
        else:
            self.ids = range(len(words))

        # Fit object to words.
        self.fit(words)

    def fit(self, words: list):
        """
        Fit a tfidf matrix to the wordlist,
        which can then be used to match
        strings back to words from the list.

        Parameters
        ---------
            words : list
                List of words for
                computing the matrix.
        """
        # Do some mild preprocessing.
        strings = list(map(lambda x: str(x).lower(), words))
        # Create the tfidf vectorizer with
        # ngram analyzer.
        self.vectorizer = TfidfVectorizer(analyzer=ngrams)
        # Fit the vectorizer to the data.
        self.matrix = self.vectorizer.fit_transform(strings)

    def get_vector(self, string: str):
        """
        Get ngram vector representation of a
        string based on the fitted vectorizer.

        Parameters
        ---------
            string : str
                String to transform.

        Returns
        ---------
            vector : csr_matrix
                Compressed sparse row matrix
                representation.
        """
        # Cast the query string into a list because sklearn vectorizer
        # wants an array-like object.
        string = [string]
        # Vectorize the string so we can match it against the matrix.
        vector = self.vectorizer.transform(string)

        return vector

    def match(self, string: str, bound: float = 0.7):
        """
        Match a query string against the list
        of fitted strings via the tfidf matrix.

        Parameters
        ---------
            string : list
                List with single string to which we wish to match strings from the
                source list.
            bound : float
                Lower bound below which we will ignore
                matched strings.

        Returns
        ---------
            matches : list[tuple]
                List of tuples containing the original string, the matched
                string, and the cosine distance between their tfidf vectors.
        """

        def cossim_top(query_matrix, ntop=10, lower_bound=0):
            """
            Returns csr matrix with topn matches to the fitted
            matrix for for a given query matrix

            Parameters
            ---------
                query_matrix : np.array
                    Vectorized matrix for which we wish to
                    identify close neighbors.
                ntop : int
                    Number of top matches to grab to speed up
                    processing.
                lower_bound : float
                    Lower distance bound below which we will
                    ignore matches.

            Returns
            ---------
                matches : csr_matrix
            """
            # Force the query and target into csr matrices. If they
            # are already, there is no overhead.
            A = self.matrix.tocsr()
            B = query_matrix.tocsr()

            return awesome_cossim_topn(A, B.transpose(), ntop, lower_bound)

        # Vectorize the string so we can match it against the matrix.
        query = self.get_vector(string)
        # Search for top matches. Ignore those which are too small.
        matches = cossim_top(query, lower_bound=bound)
        # Get all the non-zero matches.
        non_zero = matches.nonzero()
        # Get the ids of the non-zero matches to map back to words.
        match_ids = non_zero[0]
        # Return as a list of tuples.
        matches = [(self.words[j], matches.data[i], j) for i, j in enumerate(match_ids)]
        # Sort it by closest match.
        return sorted(matches, key=lambda x: x[1], reverse=True)
