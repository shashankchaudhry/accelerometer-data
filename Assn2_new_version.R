library(stats)
library(PerfMeas)
library(LiblineaR)
library(randomForest)
library(tree)

setInternet2(use=T)
fileUrl <- "https://spark-public.s3.amazonaws.com/dataanalysis/samsungData.rda"
download.file(fileUrl,destfile=".\\SamsungData.rda")
load("SamsungData.rda")
dim(samsungData)
str(samsungData)
head(samsungData)
sum(is.na(samsungData))

#shuffle the rows, take 60% to train, 20% to eval and 20% to test
set.seed(42)
shuffledSamsData <- samsungData[sample(nrow(samsungData)),]
num_rows <- dim(samsungData)[1]
num_train <- floor(num_rows* 0.8)
trainSamsIndices <- c(1:num_train)
testSamsIndices <- c((num_train+1):num_rows)

# save col names and rename cols
store_names <- names(shuffledSamsData)
names(shuffledSamsData) <- paste0(rep('c',dim(shuffledSamsData)[2]),as.character(seq(1:dim(shuffledSamsData)[2])))

# exploratory analysis
# convert all actions to numeric and try glms
glm_y <- as.factor(shuffledSamsData$c563)
glm_y <- as.numeric(glm_y)

# check the results using cross-validation on eval set
glm_eval_data <- shuffledSamsData[trainSamsIndices,c(-562,-563)]
glm_test_data <- shuffledSamsData[testSamsIndices,c(-562,-563)]
glm_eval_y <- glm_y[trainSamsIndices]
glm_test_y <- glm_y[testSamsIndices]


#Create 10 equally size folds
folds <- cut(seq(1,nrow(glm_eval_data)),breaks=10,labels=FALSE)

num_classes <- 6

# function to convert single col of classes into matrix
conv.vect.to.classmat <- function(x){
	mat <- matrix(rep(0,num_classes*length(x)), nrow=length(x), ncol=num_classes)
	for(j in 1:length(x)){
		mat[j,x[j]] <- 1
	}
	return(mat)
}

cross.validate.glm.formula <- function(frmla){
	print(frmla)
	f_vals <- rep(0,10)
	#function to perform 10 fold cross validation
	for(i in 1:10){
		#Segement your data by fold using the which() function 
		testIndexes <- which(folds==i,arr.ind=TRUE)
		testData <- glm_eval_data[testIndexes, ]
		trainData <- glm_eval_data[-testIndexes, ]
		test_y <- glm_eval_y[testIndexes]
		train_y <- glm_eval_y[-testIndexes]
		#train model on train set and test on test set
		fold_glm <- glm(as.formula(frmla), data = trainData)
		pred_y <- round(predict(fold_glm,testData))
		# reset everything < 1 to 1 and everything greater than 6 to 6
		pred_y[which(pred_y < 1)] <- 1
		pred_y[which(pred_y > 6)] <- 6
		# make a matrix of results for F.measures
		test_mat <- conv.vect.to.classmat(test_y)
		pred_mat <- conv.vect.to.classmat(pred_y)
		# set mat col names
		colnames(test_mat) <- colnames(pred_mat) <- LETTERS[1:num_classes]
		results <- F.measure.single.over.classes(test_mat, pred_mat)
		f_vals[i] <- results$average['F']
	}
	return(mean(f_vals))
}
all_fmla <- 'train_y ~ .'
all_score <- cross.validate.glm.formula(all_fmla)
# returns mean of CV F-values: 0.9000818

# using deviance to get a subset of important parameters
glmTesting <- rep(0,561)
for (i in 1:561){
	this.col <<- glm_eval_data[,i]
	glmTesting[i]<-deviance(glm(glm_eval_y~this.col,data=glm_eval_data))
}
quantile(glmTesting)
hist(glmTesting)
best_fmla <- all_fmla
best_score <- all_score
for (i in seq(from=14000, to=16000, by=500)){
	colsToKeep <- grep("TRUE",glmTesting < i)
	colsToKeep <- paste('c', colsToKeep, sep='')
	fmla <- paste('train_y ~ ', paste(colsToKeep, collapse= '+'))
	cv_score <- cross.validate.glm.formula(fmla)
	best_fmla <- if(cv_score > best_score) fmla else best_fmla
	best_score <- if(cv_score > best_score) cv_score else best_score
}
print(best_fmla)
print(best_score)
# best result in this case continues to be the one with all the cols
# "train_y ~ ."
# 0.9000818

# getting final results on test data with this model
glm_final_model <- glm(glm_eval_y ~ ., data = glm_eval_data)
pred_test_data_glm <- round(predict(glm_final_model,glm_test_data))
# reset everything < 1 to 1 and everything greater than 6 to 6
pred_test_data_glm[which(pred_test_data_glm < 1)] <- 1
pred_test_data_glm[which(pred_test_data_glm > 6)] <- 6
# make a matrix of results for F.measures
glm_test_mat <- conv.vect.to.classmat(glm_test_y)
glm_pred_mat <- conv.vect.to.classmat(pred_test_data_glm)
# set mat col names
colnames(glm_test_mat) <- colnames(glm_pred_mat) <- LETTERS[1:num_classes]
results <- F.measure.single.over.classes(glm_test_mat, glm_pred_mat)
final_f_val <- results$average['F']
print(final_f_val)
# value:0.9054036 

# trying multiclass classification using LiblineaR
# converting the data to matrix as it 
eval_matrix <- as.matrix(glm_eval_data)
test_matrix <- as.matrix(glm_test_data)
# Center and scale data
scaled_eval_mat <- scale(eval_matrix,center=TRUE,scale=TRUE)
c_val <- heuristicC(eval_matrix)

# build the model
cross_validation_val <- LiblineaR(data=scaled_eval_mat, labels=glm_eval_y, cross=10, cost=c_val, bias=TRUE, verbose=FALSE)
# cross_validation_val: 0.9767046
# making model
liblinear_model <- LiblineaR(data=scaled_eval_mat, labels=glm_eval_y, cost=c_val, bias=TRUE, verbose=FALSE)

# Scale the test data
scaled_test_mat <- scale(test_matrix,attr(scaled_eval_mat,"scaled:center"),attr(scaled_eval_mat,"scaled:scale"))
# Make prediction
p=predict(liblinear_model, scaled_test_mat)

# make a matrix of results for F.measures
liblinear_test_mat <- glm_test_mat
liblinear_pred_mat <- conv.vect.to.classmat(as.numeric(unlist(p)))
# set mat col names
colnames(liblinear_test_mat) <- colnames(liblinear_pred_mat) <- LETTERS[1:num_classes]
results <- F.measure.single.over.classes(liblinear_test_mat, liblinear_pred_mat)
final_f_val_liblinear <- results$average['F']
print(final_f_val_liblinear)
# value:0.9783832

# using a non-linear model, Random Forests:
glm_eval_y <- as.factor(glm_eval_y)
randomForest1 <- randomForest(formula = glm_eval_y~. , data = glm_eval_data , prox=TRUE)
pred_rf <- predict(randomForest1,glm_test_data)

rf_test_mat <- glm_test_mat
rf_pred_mat <- conv.vect.to.classmat(as.numeric(unlist(pred_rf)))
colnames(rf_test_mat) <- colnames(rf_pred_mat) <- LETTERS[1:num_classes]
results <- F.measure.single.over.classes(rf_test_mat, rf_pred_mat)
final_f_val_rf <- results$average['F']
print(final_f_val_rf)
# value: 0.9744006

#plotting
# 1) exploratory: some variable coloured in activity for 1 person
# 2) one tree with the text
# 3) plotting the test set data with actual colors and wrongly identified bigger

newData <- shuffledSamsData
newData$activity <- glm_y
tree1 <- tree(activity~.,data=newData)
prediction1 <- predict(tree1)

sizeFactor <- rep(0,length(prediction1))
for (i in 1:length(prediction1)){
	if (prediction1[i]!=newData$activity[i]){
		sizeFactor[i] <- 1.2
	}
	else{
		sizeFactor[i] <- 0.6
	}
}

par(mfrow=c(1,3))
plot(newData[newData$c562==1,50],newData[newData$c562==1,51],col=as.factor(newData[newData$c562==1,562]),pch=19,xlab="tGravity Acc Max in X dir", ylab="tGravity Acc Max in Y dir",cex.lab=1.5,cex.axis=1.5)
legend(0,0,legend=c("Laying","Sitting","Standing", "Walk", "Walkdown", "Walkup"),col=c(1,2,3,4,5,6),pch=19,cex=0.9, xjust=0.5)
mtext(text="(a)",side=3,line=1)
plot(tree1)
text(tree1,cex=1.0)
mtext(text="(b)",side=3,line=1)
plot(newData[,50],newData[,51],col=as.factor(newData[,562]),pch=19,xlab="tGravity Acc Max in X dir", ylab="tGravity Acc Max in Y dir",cex.lab=1.5,cex.axis=1.5,cex=as.numeric(sizeFactor))
mtext(text="(c)",side=3,line=1)
legend(0,0,legend=c("Laying","Sitting","Standing", "Walk", "Walkdown", "Walkup"),col=c(1,2,3,4,5,6),pch=19,cex=0.9, xjust=0.5)


dev.copy2pdf(file=".\\finalfigureassn2.pdf")






