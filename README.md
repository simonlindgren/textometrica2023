![textometrica](logo.png)

[*Textometrica*](https://web.archive.org/web/20120201063603/http://textometrica.humlab.umu.se/) was an application for combining quantitative content analysis, qualitative thematization, and network analysis, originally conceived by me, Simon Lindgren, and coded in PHP by Fredrik Palm at Humlab, Umeå University, in 2011.

This app, coded in Python by [Simon Lindgren](https://github.com/simonlindgren), makes the Textometrica workflow available anew. 

If you use this approach, conceived as CCA (Connected Concept Analysis), please cite:

> Lindgren, S. (2016). \"Introducing Connected Concept Analysis\". *Text & Talk*, 36(3), 341–362 [[doi](https://doi.org/10.1515/text-2016-0016)]


### Run the app on Streamlit Community Cloud

Go to [https://textometrica.streamlit.app](https://textometrica.streamlit.app)


### ... or run it locally on your computer

Manually:

- Clone this repository.
- Make sure you have streamlit installed (`pip install streamlit`).
- Run `pip install -r requirements.txt`
- Run `streamlit run app.py`



... or with docker:

- With docker installed, clone and cd into repository.
- Then:
```
$ sudo docker build -t textometrica .
$ sudo docker run -p 8501:8501 textometrica
```
- Expected result:
```
You can now view your Streamlit app in your browser.
URL: http://0.0.0.0:8501
```
- Go to `http://<ip-where-you-ran-the-docker-image>:8501`
