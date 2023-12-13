# Phase 4 Project: Movie Recommender System

## Project Team: Group 2

   1. Rose Kyalo
   2. Angel Linah Atungire
   3. Oscar Mulei
   
## Table of Contents

1. [Overview](#overview)
2. [Business Problem](#business-problem)
3. [Key Objectives](#key-objectives)
4. [Data Understanding](#data-understanding)
5. [Project Structure](#project-structure)
6. [Exploratory Data Analysis](#exploratory-data-analysis)
7. [Features](#features)
8. [Dependencies](#dependencies)
9. [Installation](#installation)
10. [Execution](#execution)
11. [Code Snippets](#code-snippets)
12. [Future Improvements](#future-improvements)
13. [Conclusion](#conclusion)
14. [License](#license)
15. [Additional Information](#additional-information)

## Overview <a name="overview"></a>

In the past, movie consumption involved purchasing physical tapes or later, flash drives, allowing access to individual movies for a price, often around Ksh 30 to Ksh 50 per movie. However, with the rise of streaming platforms like Netflix, Hulu, and Amazon Prime Video, the landscape has drastically changed. Users now have access to an extensive library of content through subscription models, providing an array of movies and shows for a fixed fee or even for free in certain cases.You might have noticed that when you finish watching a movie or show on Netflix, the platform might suggest similar titles in terms of genre, actors, directors, or themes. These are recommender systems that personalize the user experience by offering relevant content, ultimately keeping users engaged and increasing the likelihood of them finding content they'll enjoy. This not only enhances user satisfaction but also contributes to longer user retention on the platform.

## Business Problem <a name="business-problem"></a>

SilverScreen Studios, a leader in the movie production industry, is focused on optimizing its promotional strategies for its diverse film portfolio. The company has sought our expertise in engineering a robust movie recommendation system. This system is intended to deliver bespoke movie suggestions to its audience, thereby augmenting user engagement and supporting successful promotional campaigns.

## Key Objectives <a name="key-objectives"></a>

The principal objective of this system is to analyze and understand user behaviors and preferences through their movie rating history. With this insight, the system will strive to:

**1. Precision in Recommendations**: Develop an algorithm to accurately suggest the top five movies based on user ratings, aligning closely with individual preferences.

**2. Enhancement of User Engagement**: Create a recommendation system that significantly boosts user interaction by delivering personalized movie suggestions.

**3. Generation of Personalized Recommendations**: Tailor recommendations to align with each user's distinct interests.

## Data Understanding <a name="data-understanding"></a>

This comprehensive dataset includes:

**1. User ratings**: A collection of movie ratings provided by users, which is the cornerstone of our content based model.

**2. Movie details**: Information on various movies, including genres, release dates, and more, which aids in understanding the context of user preferences.

**3. Links**: References to other databases, which could be useful for enriching our dataset with additional movie information.

**4. Tags**: User-generated tags for movies, offering insights into the nuanced preferences of users.

These files collectively furnish a comprehensive view of user interactions with movies, encompassing both quantitative ratings and qualitative descriptive tags, offering a rich source of data for our recommendation system.

## Project Structure <a name="project-structure"></a>

**1. Data Collection & Preprocessing**: Acquire and clean the MovieLens dataset.

**2. Exploratory Data Analysis**: Conduct statistical and visual analysis to identify patterns and trends.

**3. Model Development**: Contains detailed steps for building content-based recommendation systems.

**4. Interactive Widgets**: Demonstrates the implementation of interactive widgets for user interaction.

**5. User Preference Identification**: Identifies and recommends movies based on user similarity.

**6. Genre-Based Recommendations**: Recommends movies based on user-provided genres.
Usage

## Exploratory Data Analysis <a name="exploratory-data-analysis"></a>

#### Distribution of ratings
Majority of movies received ratings of 4 and 3. Conversely, a smaller number of movies were rated at 0.5, indicating that very few movies garnered such low ratings.

![png](/images/output_16_0.png)

#### Genre Analysis
The most common genres are Drama and Comedy, indicating diverse user preferences.

![png](/images/output_16_1.png)

#### Top Watched Movies 

![png](/images/output_16_2.png)

#### Average Movie Rating by Release Year
We plotted a scatter plot to ascertain a potential relationship between a movie's release year and its average rating.It revealed a distinct clustering pattern, showcasing average ratings predominantly within the range of 2 to 4.5 for movies aged from 0 to around 50 years. This suggests a tendency for recently released movies to accumulate ratings within this particular span. However, as movies surpass the 60-year mark, the clustering diminishes notably. This trend implies a shift in audience interest towards newer iterations or fresher content as movies age, leading to decreased clustering and diversity in ratings for older movies.

![png](/images/output_16_3.png)

## Features <a name="features"></a>

**1. Content-Based Recommendation**: Recommends movies based on similarities in movie titles and genres.

**2. Interactive Search Tool**: Allows users to input movie titles and receive recommendations instantly.

**3. User Similarity Identification**: Identifies users with similar movie preferences.

**4. Genre-Based Recommendation**: Recommends movies based on user-provided genres.


## Dependencies <a name="dependencies"></a>

├── pandas
├── numpy
├── matplotlib
├── seaborn
├── scipy
├── sklearn
├── ipywidgets

## Installation <a name="installation"></a>

To get started with the project, follow these steps:

**1. Fork and Clone the Repository**

Fork the repository to your GitHub account, and then clone it to your local machine using the following command:

bash
Copy code
git clone https://github.com/omulei/Movie_Recommender_System.git

**2. Access the Jupyter Notebook**

Navigate to the cloned repository directory:

bash
Copy code
cd Your-Repository

Open the Jupyter Notebook file to access the code:

bash
Copy code
jupyter notebook Your-Notebook.ipynb

**3. Check Dependencies**

Ensure you have all the required dependencies installed. You can find the necessary libraries and their versions in the requirements.txt file:

bash
Copy code
pip install -r requirements.txt

This will install all the required libraries to run the project.

## Code Snippets <a name="code-snippets"></a>


<!-- markdown cell -->

#### Movie Recommendation Widget

Create a text input widget for entering the movie title.

```python
import ipywidgets as widgets

# Create a text input widget for entering the movie title.
movie_name_input = widgets.Text(
    value='Toy Story',
    description='Movie Title:',
    disabled=False
)

# Create an output widget for displaying movie recommendations.
recommendation_list = widgets.Output()

# Define a function to trigger recommendations when text is typed.
def on_type(data):
    with recommendation_list:
        recommendation_list.clear_output()
        title = data["new"]
        # Check if the entered movie title is sufficiently long.
        if len(title) > 5:
            # Search for movie titles that match the entered text.
            results = search(title)
            # Get the movie ID of the first matching result.
            movie_id = results.iloc[0]["movieId"]
            # Display recommended movies based on the entered movie.
            display(find_similar_movies(movie_id))

# Observe changes in the text input and trigger recommendations.
movie_name_input.observe(on_type, names='value')

# Display the movie title input and the recommendation list.
display(movie_name_input, recommendation_list)
```



Future Improvements <a name="future-improvements"></a>

**1. Real-Time Integration**: Plan to integrate real-time data for current trends.

**2. Enhanced User Data**: Incorporate more user data for detailed and accurate recommendations.

**3. Platform Deployment**: Consider deployment on web/mobile platforms for broader accessibility.

Conclusion <a name="conclusion"></a>

Our movie recommendation system leverages user behaviors and preferences to deliver precise, personalized movie suggestions. Through advanced algorithms, we ensure top-rated recommendations aligned with individual tastes, fostering enhanced user engagement. The system's adaptability and user-centric approach signify its potential for longer user retention and successful promotional strategies. This project marks a pivotal step in optimizing user experiences within the entertainment industry, paving the way for future enhancements and broader accessibility.

License <a name="license"></a>

This project is licensed under the MIT License - see the LICENSE file for details.

Additional Information <a name="additional-information"></a>

Please refer to the detailed documentation and code snippets provided in this repository for a deeper understanding of the project's functionalities and implementation.

