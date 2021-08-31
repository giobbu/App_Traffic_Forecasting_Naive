# App for Naive Traffic Forecasting

## Build Image from Dockerfile

To create an image first:

```{r}
git clone https://github.com/giobbu/App_Traffic_Forecasting
cd App_Traffic_Forecasting
```

Then run:
```{r}
docker build -t naiv_traff_app .
```

Check the image created:
```{r}
docker image list
```

## Run Streamlit App Container
To interact with the App type:
```{r}
docker run -p 8501:8501 --rm naiv_traff_app
```
view your Streamlit app in your browser
```{r}
localhost:8501
```

