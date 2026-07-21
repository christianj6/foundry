class Entity:
    """
    Object to store relevant
    information for a single
    identified entity that this
    can be easily extracted for
    outputs.
    """

    def __init__(
        self,
        page: int,
        location: "range",
        string: str,
        match: str,
        idx: str,
        score: float,
        text: list,
        environment: str = None,
    ):
        """
        Instantiate object and parse the text
        to format the entity's environment ie
        the surrounding text.

        Parameters
        ---------
            page : int
                Page number.
            location : range
                Span of token integers
                which define the entity.
            string : str
                Original string of text.
            match : str
                Matched text.
            idx : int
                Index of matched text if
                this is a relevant metavariable.
            score : float
                Match confidence as
                cosine distance.
            text : list
                Source text used to reconstruct the
                environment.
            environment : str
                Formatted string containing
                the text around the matched entity.
        """
        # Assign miscellaneous attributes.
        self.page = page
        self.location = location
        self.string = string
        self.match = match
        self.idx = idx
        self.score = score
        self.is_invalid = 0
        # Format environment from the text.
        self.environment = self.format_environment(text, self.location)

    @staticmethod
    def format_environment(
        text: list,
        location: "range",
    ) -> str:
        """
        Formats a string representing the
        environment surrounding the token.

        Parameters
        ---------
            text : list
                Original text as
                list of tokens.
            location : range
                Span of text where
                the entity occurs.

        Returns
        ---------
            environment : str
                Formatted string
                representing the entity
                within its
                environment.
        """
        environment = ""
        # Extract only the text around the entity.
        text = text[location[0] - 10 : location[-1] + 11]
        # Reset the location to this slice.
        location = range(10, 10 + len(location))
        # Build a string by iterating through tokens.
        for i, token in enumerate(text):
            if i in location:
                token = token.upper()

            environment += token + " "

        return environment
