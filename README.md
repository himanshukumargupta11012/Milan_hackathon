# Canteen Recommendation & Review System - IIT Hyderabad

Welcome to the Canteen Recommendation & Review System for IIT Hyderabad! This web application offers personalized food recommendations and insights based on user feedback. Users can also search for specific items available in the canteen. Additionally, the canteen administrator can access an admin page to view user reviews and work on improving the food items.

## Features

- <ins>**Feedback Collection**</ins>: Users can provide feedback on the canteen's food.

- <ins>**Search Functionality**</ins>: Users can search for any item available in the canteen.

- <ins>**Insightful Recommendations**</ins>: Our system leverages advanced ML models, including collaborative filtering, sentiment analysis, and aspect-based sentiment analysis , to generate personalized food recommendations.

- <ins>**Aspect Sentiment Triplet Extraction**</ins>: Utilizing pyabsa, this system breaks down text data into aspects (e.g., product features), extracts opinions about those aspects, and assigns sentiment polarities (positive, negative) to each opinion, providing insights into consumer feedback or reviews.

- <ins>**Search Recommendation**</ins>: Implemented search recommendations for food items to enhance user experience using String matching algorithm.

- <ins>**Sentiment Analysis with Fine-tuned BERT Model**</ins>: fine-tuned BERT model for sentiment analysis. The system accurately analyzes textual data and predicts ratings or sentiment scores out of 5.

- <ins>**Collaborative Filtering**</ins>: Utilizing the SVD technique, the system offers recommendations based on collaborative filtering by considering user reviews on food items and recommending similar food choices from other users with shared interests.

- <ins>**Summarization using Gensim**</ins>: Gensim aids in summarizing extensive feedback, providing concise overviews of user sentiments and preferences.

- <ins>**Keyword Extraction with Gensim**</ins>: Gensim assists in extracting key terms and phrases from feedback, enabling a deeper understanding of the main aspects of user opinions.

- <ins>**Flask-based Website**</ins>: The backend of this system is powered by Python and Flask, utilizing Jinja templates for rendering HTML, CSS, and JavaScript, enhancing the user interface and displaying the results seamlessly.


## Contributors

- [Himanshu Kumar Gupta](https://github.com/himanshukumargupta11012) - Project Lead
- [Gunjit Mittal](https://github.com/gunjitmittal) 
- [Donal Loitam](https://github.com/Donal-08) 
- [Suraj Kumar](https://github.com/kumarsuraj151) 
- [Ravula Karthik](https://github.com/karthik6281) 
- [Mannem Charan](https://github.com/charanyash) 

## Installation

1. Clone the repository:
```bash
git clone <repository_url>
 ```


2. Install the required dependencies:
```bash
pip install -r requirement.txt
```

3. Install the required dependencies:
```bash
cd backend/
flask run 
```

4. Access the website:
Open a web browser and navigate to `http://localhost:5000` to access the Canteen Recommendation System.
