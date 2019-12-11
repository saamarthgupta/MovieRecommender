from django.shortcuts import render_to_response
from django.http import HttpResponse, HttpResponseRedirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.template.loader import get_template
from django.template import RequestContext, Context
from .forms import RegistrationForm
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import os
from django.conf import settings

#import imdb
#import itertools
from django.contrib.auth.decorators import login_required
# Create your views here.
def index(request):
	variables = Context({
		'user': request.user,
		})
	return render_to_response('index.html', variables)

def login_user(request):
	template = 'login.html'
	state="Please fill in your credentials"
	log = False
	incorrect=False
	if request.POST:
		username = request.POST.get('username')
		password = request.POST.get('password')
		user = authenticate(username=username, password=password)
		if user is not None:
			if user.is_active:
				login(request, user)
				state = "Logged in!"
				log = True
				return HttpResponseRedirect('/')
			else:
				state = "Not registered user"
		else:
			state = "Incorrect username or password!"
			incorrect = True
	variables = Context({
		'state': state,
		'log': log,
		'incorrect' : incorrect
		})
	return render_to_response(template, variables, RequestContext(request) )

def logout_user(request):
	logout(request)
	return HttpResponseRedirect('/') 

@login_required
def search(request):
	query = "No Results Found"
	sent=False
	recommended = []
	results = []
	
	variables = []
	
	file_ = open(os.path.join(settings.BASE_DIR, 'finalCleanData.csv'))
	movies=pd.read_csv(file_,delimiter=',')
	if request.GET:
		query = request.GET['query']
		if(query==""):
			showResults=False
			variables = Context({'show_results' : showResults})
			return render_to_response('search.html',variables, RequestContext(request))

		tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 1),min_df=0, stop_words='english')
		sent=True
		tfidf_matrix = tf.fit_transform(movies['genres'])
		cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
		titles = movies['originalTitle']
		indices = pd.Series(movies.index, index=movies['originalTitle'])
		
		def genre_recommendations(title):
			
			# idx = indices[title]
			idx = movies[movies['originalTitle'].str.contains(title, case=False)]
			empty=False
			message = ""
			recommended = []
			mainResult=[]
			showResults=False
			if(idx.empty):
				
				message = "No Matching Results Found."
				empty=True
				
			else:
				
				i = idx.to_numpy()
				count = 1
				for p in range(i.shape[0]):
					if (count>5):
						break
					result = {'id' : i[p][0], 'originalTitle' : i[p][3] , 'year' : i[p][5] }
					mainResult = i[0][3]
					# print("Result : ",result)
					results.append(result)
					count=count+1
				
				showResults=True
				# print(idx)
				#idx = idx.iloc[0].index
				idx=indices[idx.iloc[0]['originalTitle']]
				# print(movies.iloc[idx])
				x=cosine_sim[idx]
				x=np.delete(x,idx,0)
				sim_scores = list(enumerate(x))
				sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
				sim_scores = sim_scores[0:21]
				movie_indices = [i[0] for i in sim_scores]
				recommend = movies.iloc[movie_indices] 
				
				temp = recommend.to_numpy()

				for p in range(temp.shape[0]):
					rec = {'id' : temp[p][0], 'originalTitle' : temp[p][3] , 'year' : temp[p][5]}
					recommended.append(rec)
				
				# print("Results : ",results)
				# print("Recommended : ", recommended)

			variables = Context({
				#'mvies': mvies,
				# 'posters': posters,
				'recommended': recommended,
				'results': results,
				'show_results': showResults,
				'empty' : empty,
				'query' : query,
				'message' : message,
				'mainResult' : mainResult
			})
			return variables
	

		#access = imdb.IMDb()
		#	titles = []
		#	covers = []
		#	for i in results:
		#		movie_id = links.objects.get(movie_id=i.movie_id)
		#		movie = access.get_movie(str(movie.imdb_id))
		#		titles.append(movie['title'])
		#		covers.append(movie['cover url'])
		
		variables = genre_recommendations(query)

	
	# print("Final Response : ",variables)
	return render_to_response('search.html',variables, RequestContext(request))

def main(request):
	return render_to_response('main.html')

def reg_user(request):
	state= ""
	template = 'register.html'
	if request.POST:
		form = RegistrationForm(request.POST)
		if form.is_valid():
			user = User.objects.create_user(
				username = form.cleaned_data['username'],
				password = form.cleaned_data['password1'],
				email = form.cleaned_data['email']
			)
			return HttpResponseRedirect('/login')
		else:
			state = "Incorrect Credentials!"
	else:
		form = RegistrationForm()
	variables = RequestContext(request, {'form':form, 'state':state})
	return render_to_response('register.html', variables)

def movie(request):
	template = 'movie.html'
	query = ""
	if request.GET:
		query = request.GET['query']
		access = imdb.IMDb()
		movie = access.get_movie(str(query))

	variables = Context({
			'movie': movie
		})
	return render_to_response(template, variables)
