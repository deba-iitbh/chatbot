# Chatbot [NLP Project]

## Dataset
We have created a custom dataset which includes the FAQ of IIT Bhilai. 

### Collection Procedure
Scraped different websites like Quora, College-Dunia, etc.

## Frontend
Its made using [django](https://www.djangoproject.com/) framework.
To start the server. Please run
```shell
source chating/.env
python manage.py runserver
```

## Backend
### Baseline Model
We have used word vectors (Word2Vec, FastText) to determine the embedding of different questions, and then tried
to find the Word Mover's distance between the asked question and available questions, to give ans answer.

Members - 
1. [Satyam](https://github.com/satyams2812)
2. [Shahid](https://github.com/sowdagar3)
3. [Debarghya](https://github.com/deba-iitbh)
4. [Nikhil](https://github.com/nikhildotpy)
