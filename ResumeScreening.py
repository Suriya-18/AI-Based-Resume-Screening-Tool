import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained NLP model
nlp = spacy.load('en_core_web_sm')

# Function to extract skills from resume
def process_resume(resume_text):
    doc = nlp(resume_text)
    skills = ["python", "machine learning", "data analysis", "AI", "SQL", "deep learning", "natural language processing"]
    matches = []
    for skill in skills:
        if skill in resume_text.lower():
            matches.append(skill)
    return matches

# Function to compare resume with job description
def match_resume_to_job(resume_text, job_description):
    cv = CountVectorizer()
    all_texts = [resume_text, job_description]
    count_matrix = cv.fit_transform(all_texts)
    similarity_matrix = cosine_similarity(count_matrix[0:1], count_matrix)
    return similarity_matrix[0][1]

def main():
    resume = input("Enter resume text: ")
    job_description = input("Enter job description text: ")
    
    skills_matched = process_resume(resume)
    print(f"Skills matched: {', '.join(skills_matched)}")
    
    similarity_score = match_resume_to_job(resume, job_description)
    print(f"Resume match score with job description: {similarity_score:.2f}")

if __name__ == "__main__":
    main()
