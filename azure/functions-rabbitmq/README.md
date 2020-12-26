# Create a Simple Text Processing Service with Azure Functions and RabbitMQ

***

### Overview

Suppose one designs a text processing application where customers submit raw, tabular data for analysis. If the analysis demands several hours of computation time, it is advisable to first validate the submitted data to ascertain its compatibility with the rest of the application pipeline, lest the customer encounter frustrating application errors.

For development agility it is recommended to keep this quick, preliminary data validation module separate from other services. However, provisioning a virtual machine for this small operation is overkill.

[Azure Functions](https://azure.microsoft.com/en-us/services/functions/) is a sensible offering for small operations which 'glue' together components of a larger application infrastructure. Functions can be provisioned on a consumption-based, serverless plan which only bills for the triggered computation time of the function, making these resources potentially very economical.

In this tutorial, we will:

1. Create a simple data validation module.
2. Create an Azure Functions application and corresponding Storage Account File Share.
3. Integrate the data validation module with our Function such that it is triggered when a new file is added to the File Share.
4. Complete our Function with a conditional HTTP response providing metadata on the file's compatibility.
5. Evaluate the costs of our Function.

***

### 1. Create a Simple Data Validation Module

This tutorial does not focus on NLP development, so I will use a dummy script to demonstrate the capabilities of the Functions app. This function validates the language of a .csv file, which may be important if an analysis pipeline assumes a limited set of languages or has language-dependent functions. One could, of course, introduce any reasonably-sized validation method.

The [snippets](...) directory contains the file [language_detection.so](...), which is a Linux-compatible compiled library file. Because we will host the Azure Function on Linux, we must use this file format.

Next we create a simple script with a `main()` method which will become our Function entry point. Azure Functions expects this in an `__init__.py`  file. Do not worry about creating a directory structure for now, we will begin with the script and then insert it into the directory structure Azure Functions makes for us.

*`__init__.py`*

```output
from language_detection import detect_lang


def main(trigger):
	pass
```

`detect_lang` is a simple language detection which accepts a string of text and returns a dictionary containing the detected language in a "lang" entry. The argument `trigger` will be the Function trigger configured by Azure Functions. Our entry point will parse this trigger, access a file, and return a message detailing the languages detected in the file.

We will ultimately configure our Function to accept an HTTP request notifying the application when a new file is uploaded to storage, and providing the location of the file in the File Share. The Function will return an HTTP response with our message. Thus, we need to add functionality for processing an HTTP request and sending a response. We will also add logging functionality.

Suppose our HTTP request is a JSON with the following format:

```output
{
    "directory": "",
    "filepath": ""
}
```

where `directory` is the folder of the new .csv file, and `filepath` is the name of the file. 

Suppose we wish to send a response with the following parameters:

```output
{
	"message": "",
	"n_comments": "",
	"n_words": "",
	"languages": "",
	"sample_comment": ""
}
```

where `message` is a conditional message for error-handling, `languages` is a list of the languages detected by our canned function, and `sample_comment` is an example comment from the dataset. With these metadata our imaginary application is provided immediate feedback on the compatibility of the dataset.

Integrating the input and output with our example script, we obtain the following:

```python
from language_detection import detect_lang
import azure.functions as func
import logging
import json


def main(trigger:func.HttpRequest)->func.HttpResponse:
	logging.info("HTTP Trigger successful.")
	logging.info("Getting directory ...")
    # Attempt to identify the directory.
    directory = trigger.params.get("directory")
    # Error handling if this initial get method fails.
    if not directory:
        try:
            body = trigger.get_json()
        except ValueError:
            pass
        else:
            directory = body.get("directory")
            
    logging.info("Getting filepath ...")
    # Attempt to identify the filepath.
    filepath = trigger.params.get("filepath")
    # Error handling if this initial get method fails.
    if not filepath:
        try:
            filepath = body.get("filepath")
        except ValueError:
            pass
        
    if directory and filepath:
        # TODO: Access the file and perform data validation.
		pass
    
    else:
        response = {
            "message": "The request lacks necessary parameters.",
            "n_comments": None,
            "n_words": None,
            "languages": None,
            "sample_comment": None
        }
        
        return func.HttpResponse(
        	json.dumps(response),
            status_code=400,
            mimetype="application/json"
        )
```

`azure.functions` is the Azure library containing the relevant objects `func.HttpRequest` and `func.HttpResponse` which constitute our input and output, respectively. After providing a few logging messages, we attempt to access the `directory` and `filepath` necessary to retrieve the uploaded file. If these parameters are not available, we immediately return an error response, which must be formatted into a string with the `json` library. The keyword argument `mimetype` reflects that our response is a JSON string, and the `status_code` tells our client that the request was invalid.

Next, we need to connect to the file share when provided with the relevant parameters and process the .csv file. We will process the .csv with the `pandas` library and our canned language detection function. Later, when we create the file share on Azure, it will become clearer how the connection operates, but for now it is sufficient to know that it operates like a standard file directory.

Connecting to the file share of an Azure Storage Account requires a `storageAccountName` and `storageAccountKey` which are retrieved from Azure and which we will store in environment variables. Therefore, we connect to the file share  via the environment variables.

```python
from azure.storage.file import FileService
import os

file_service = FileService(account_name=os.environ['storageAccountName'], 		
                           account_key=os.environ['storageAccountKey'])
```

Later, when we configure the Azure Function application settings, it is those settings which will be accessed as the environment variables.

Once a connection to the file share is established, we use the `directory` and `fileshare` to retrieve the relevant file as text and cast this into a pandas `DataFrame`. We can then return the relevant metadata we originally intended.

```python
from azure.storage.file import FileService
from io import StringIO
import pandas as pd
import os

file_service = FileService(account_name=os.environ['storageAccountName'], 		
                           account_key=os.environ['storageAccountKey'])
file = file_service.get_file_to_text(os.environ['fileShareName'],
                                     directory,
                                     filepath)
text = file.content
stream = StringIO(text)
df = pd.read_csv(stream)
first_column = df.columns[0]
languages = []
for entry in df[first_column]:
    lang_dict = detect_lang(entry)
    languages.append(lang_dict["lang"])
    
response = {
    "message": "The file was successfully validated.",
    "n_comments": len(df),
    "n_words": len([word for comment in list(df[first_column].values) for word in comment.split()]),
    "languages": set(languages),
    "sample_comment": df[first_column].values[0]
}
```

Integrating this functionality with our original `main()` method, including some additional error handling and logging at each stage of the processing, we obtain the final entry point to our function which can be viewed at [\__init__.py](...). We are now ready to set up the Function and Storage Account on Azure.

- [ ] Clean up the init.
- [ ] Add the rabbitMQ integration and adapt the tutorial for this.

### 2. Create an Azure Functions application and corresponding Storage Account File Share.

- [ ] Just create the instances.
- [ ] Also add creating rabbitMQ instance here.

***

### TODO

- [ ] requirements.txt
- [ ] other files necessary for the function
- [ ] organize all the keys and stuff
- [ ] update application settings and all stuff on azure portal
- [ ] then go to VSCode, integrate all the files and run the function
- [ ] deploy it, and it should work