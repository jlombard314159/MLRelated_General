setwd("C:/users/jlombardi/Documents/GitLabCode/nnplay/Classifier Pipeline/Data/")

dataFiles = list.files("C:/users/jlombardi/Documents/GitLabCode/nnplay/Classifier Pipeline/Data/",
	full.names = T,recursive = T,pattern = ".rds")

dataList <- list()

for(i in 1:length(dataFiles)){
	
	dataList[[i]] <- readRDS(dataFiles[i])
	
	dataList[[i]]$Species <- strsplit(basename(dataFiles[i]),"_")[[1]][1]
	
	if(i %%2 == 0) dataList[[i]]$DatasetType <- "Validation"
	else dataList[[i]]$DatasetType <- "Training"
	
}

#some manual coding
dataList[[3]]$LANO_KLM <- dataList[[4]]$LANO_KLM <- NULL

finalData <- do.call(rbind.data.frame,dataList)

write.csv(finalData,"AllBatCallData.csv",row.names=FALSE)

#------------------------------------------------------------------
#Test with fat datadataList[[i]] <- readRDS(dataFiles[i])
	
temp <- readRDS(dataFiles[9])
	
temp$Species <- paste0(strsplit(basename(dataFiles[9]),"_")[[1]][1],
	"_","FATS")


write.csv(temp,"SHBA_PH_FATS.csv",row.names=FALSE)

#-------------------------------------------------------------------
#Read in some other data and copy beccas code to get probabilities

gitPath <- "C:/Users/jlombardi/Documents/GitLabCode/pilothillandkellycreek/"

# Define analysis project ('Kelly Creek', 'Pilot Hill', 'Combined')
project <- 'Combined'

subFolder <- ifelse(project == 'Kelly Creek', 'KellyAnalysis',
                    ifelse(project == 'Pilot Hill', 'PilotAnalysis', 'Combined'))

dataPath <- paste0(gitPath,
                   "AcousticModeling/Data/",
                   gsub(" ", "", project),
                   "/Input 0731-1015/")

workSpacePath <- paste0(gitPath,
                        "AcousticModeling/Workspaces/", subFolder, "/")

outputPath <- paste0(gitPath, 'AcousticModeling/Output/', subFolder, '/')


#Load data
holdSpeciesInfo = "All"

speciesList <- c("All","LABO_KLM_ERBAKelly_ERBA_",
	"LACI_KLM_HOBA","LANO_KLM_SHBA")

tempList <- list()

for(i in 1:length(speciesList)){
	trainDat <- readRDS(file.path(workSpacePath, paste0(speciesList[i],'trainDat.rds')))
	validDat <- readRDS(file.path(workSpacePath, paste0(speciesList[i],'validDat.rds')))
	modelCovariates <- readRDS(file.path(workSpacePath, paste0(speciesList[i], 'ModelParameters.rds')))
	
	changeThese <- names(trainDat)[grepl("StationPressure", names(trainDat))]
	trainDat[,changeThese] <- trainDat[,changeThese]*33.8639
	
	changeHere <- names(validDat)[grepl("StationPressure", names(validDat))]
	validDat[,changeHere] <- validDat[,changeHere]*33.8639
	
	
	selectFit <- glm(modelCovariates,
	            family = 'binomial', data = trainDat)
	validDat$pred = predict(selectFit, newdata = validDat, type = 'response')
	
	
	##Prepare ROC figure
	preds = seq(0, max(validDat$pred, na.rm = T), length.out = 100)
	##truePositive is proportion of true positives as a function of predicted probability
	truePositive = sapply(preds, function(p, validDat){
	  mean(validDat$pred[validDat$BinaryResponse == 1] >= p, na.rm = T)
	}, validDat)
	##falsePositive is proportion of false positives as a function of predicted probability
	falsePositive = sapply(preds, function(p, validDat){
	  mean(validDat$pred[validDat$BinaryResponse == 0] >= p, na.rm = T)
	}, validDat)
	
	# browser()
	# 	plot(truePositive ~ falsePositive, pch = 16,
	#      main = "Kelly Creek and Pilot Hill All Bats;\n Prediction Performance (more than 1 calls)",
	#      #main = "Model performance with 0-call threshold", 
	#      xlab = 'Probability of false positive prediction',
	#      ylab = 'Probability of true positive prediction', type = 'l', lwd = 2,
	#      cex.lab = 1.5, cex.axis = 1.5)
	# abline(h = pretty(range(truePositive)), col = 'grey')
	# segments(x0 = 0, x1 = 1, y0 = 0, y1 = 1, lty = 'dashed', col = 'lightgrey', lwd = 2)
	# points(falsePositive[max(which(truePositive >= 0.7))],
	#        truePositive[max(which(truePositive >= 0.7))], pch = 16, col = 'blue', cex = 2)
	# text(falsePositive[max(which(truePositive >= 0.7))],
	#      truePositive[max(which(truePositive >= 0.7))],
	#      paste0('P(curtail | bats) = ',
	#             round(truePositive[max(which(truePositive >= 0.7))],2),
	#             '\nP(no bats | curtail) = ',
	#             round(falsePositive[max(which(truePositive >= 0.7))],2)),
	#      adj = c(0.021, 1.25), cex = .95)
	# 
	tempList[[i]] <- data.frame(preds,truePositive,falsePositive,
		speciesList[i])

}

tempDF <- do.call(rbind.data.frame,tempList)

colnames(tempDF) <- c("Prob","truePositive","falsePositive","species")
tempDF$Model <- "ElasticNet"

write.csv(tempDF,"PHKH_ROC.csv",row.names=FALSE)
