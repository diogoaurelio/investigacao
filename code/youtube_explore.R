## Starting to explore Youtube Vloggers dataset
# Source: https://www.idiap.ch

MyPath = "~/Documents/Development/phd/code"
setwd(MyPath)

## Youtube personality datasets
youtube_data_path = "~/Documents/Development/phd/data/youtube_personality/"
youtube_transcripts_path = paste(youtube_data_path, sep="/", "transcripts")
# intial 3 datasets
audiovisual_features_file = paste(youtube_data_path, "YouTube-Personality-audiovisual_features.csv", sep="/")
gender_file = paste(youtube_data_path, "YouTube-Personality-gender.csv", sep="/")
impression_scores_file = paste(youtube_data_path, "YouTube-Personality-Personality_impression_scores.csv", sep="/")

# read.csv(impression_scores_file, stringsAsFactors = F) 
impression_scores = read.csv(impression_scores_file, stringsAsFactors = F) 
gender = read.csv(gender_file)
# Header as False to process it later
audiovisual_features = read.csv(audiovisual_features_file, header = F, stringsAsFactors = F)

str(impression_scores)
head(impression_scores)
# From readme:
#The aggregated Big-Five scores are reliable with the following intra-class correlations (ICC(1,k), k=5): 
#  - Extraversion (ICC = .76), 
#- Agreeableness (ICC = .64), 
#- Conscientiousness (ICC = .45), 
#- Emotional Stability (ICC = .42), 
#- Openness to Experience (ICC = .47), 
#all significant with p < 10^{-3}.

str(audiovisual_features)
colnames(audiovisual_features)
# check first observation
audiovisual_features[1,] # is the title as expected because of Header=FALSE

# check second observation
audiovisual_features[2,]

# Clean up phase
audio_colnames = strsplit(audiovisual_features[1,], ' ') 
audio_colnames = as.vector(audio_colnames[[1]])

#before = audiovisual_features # backup
audiovisual_features2 = strsplit(audiovisual_features[,1], ' ')
typeof(audiovisual_features2)

audiovisual_features2 = as.data.frame(audiovisual_features2)
str(audiovisual_features2)

dim(audiovisual_features2) # 26 | 405 -> it needs to be transposed!!
audiovisual_features3 = as.data.frame(t(audiovisual_features2))
typeof(audiovisual_features3)
dim(audiovisual_features3)
colnames(audiovisual_features3)
colnames(audiovisual_features3) = audio_colnames
colnames(audiovisual_features3)
str(audiovisual_features3)
# remove first row unnecessary
av_final = audiovisual_features3[-1,]
str(av_final)
av_final$vlogId <- as.character(av_final$vlogId)
# coerce all other columns into numeric
#av_final$mean.pitch <- as.numeric(av_final$mean.pitch)
av_final[,2:26] = lapply(av_final[,2:26], as.numeric)
summary(av_final)
summary(av_final$mean.pitch)

# OK, finally cleaned; lets see gender
str(gender)
head(gender)
# pretty streight forward;
# lets join audiovisual with gender, to add another feature

