import pickle


datapoint = {
  "lead_source": "paid_ads",
  "industry": "technology",
  "employment_status": "employed",
  "location": "north_america",
  "number_of_courses_viewed": 2,
  "annual_income": 79276.0,
  "interaction_count": 4,
  "lead_score": 0.41
}






with open('pipeline.bin', 'rb') as f_in:
    model = pickle.load(f_in)





churn=model.predict_proba(datapoint)[0, 1]
print(churn)
