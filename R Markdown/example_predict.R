library(reticulate)
use_virtualenv("venv")
source_python("predict.py")
## check predict.py and models/ folder exists
## running this line would give you two functions:
## getFeatures(course_number) and predict(course_number, df)

getFeatures(36225)
getFeatures(36226)
getFeatures(36401)
getFeatures(36402)
getFeatures(15122)
getFeatures(15351)

features <- getFeatures(15122)
features

input <- c(2, "STAMACH", 42, "C", "C", 2.60, 2.20)
input_df <- data.frame(matrix(nrow=1, data=input))
colnames(input_df) <- features
input_df

model_output <- predict(15122, input_df)
predicted_grade <- model_output[[1]]
predicted_grade_prob <- model_output[[2]]*100
barplot(predicted_grade_prob, names.arg=c("A","B","C","D"), xlab="Grade", ylab="Probability (%)")
