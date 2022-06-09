function [train_accuracy,test_accuracy]=RVFL_train_val(TrainingData_File, TestingData_File,option)
%%%%%%%%%%% Load training dataset
train_data=load(TrainingData_File);
trainY=train_data(:,1);
trainX=train_data(:,2:size(train_data,2));
clear train_data;                                   %   Release raw training data array

%%%%%%%%%%% Load testing dataset
test_data=load(TestingData_File);
testY=test_data(:,1);
testX=test_data(:,2:size(test_data,2));
clear test_data;   




% This is the  function to train and evaluate RVFL for classification
% problem.
% N :      number of hidden neurons
% bias:    whether to have bias in the output neurons
% link:    whether to have the direct link.
% ActivationFunction:Activation Functions used.   
% seed:    Random Seeds
% option,mode     1: regularized least square, 2: Moore-Penrose pseudoinverse
% RandomType: different randomnization methods. Currently only support Gaussian and uniform.
% Scale    Linearly scale the random features before feedinto the nonlinear activation function. 
%                 In this implementation, we consider the threshold which lead to 0.99 of the maximum/minimum value of the activation function as the saturating threshold.
%                 Scale=0.9 means all the random features will be linearly scaled
%                 into 0.9* [lower_saturating_threshold,upper_saturating_threshold].
%Scalemode Scalemode=1 will scale the features for all neurons.
%                 Scalemode=2  will scale the features for each hidden
%                 neuron separately.
%                 Scalemode=3 will scale the range of the randomization for
%                 uniform diatribution.

% This software package has been developed by Le Zhang(c) 2015
% based on this paper: A Comprehensive Evaluation of Random Vector Functional Link Neural Network Variants
%  For technical support and/or help, please contact Lzhang027@e.ntu.edu.sg
% This package has been downloaed from https://sites.google.com/site/zhangleuestc/


    N=100;

    bias=false;

    link=true;

    ActivationFunction='sig';

    seed=0;


    RandomType='Uniform';

    mode=2;
  

    Scale=1;
   

    Scalemode=1;

% rand('state',seed);
% randn('state',seed);
U_trainY=unique(trainY);
nclass=numel(U_trainY);
trainY_temp=zeros(numel(trainY),nclass);
% 0-1 coding for the target 
for i=1:nclass
         idx= trainY==U_trainY(i);
        
         trainY_temp(idx,i)=1;
end
[Nsample,Nfea]=size(trainX);
N=N;
if strcmp(RandomType,'Uniform') 
    if Scalemode==3
         Weight= Scale*(rand(Nfea,N)*2-1);
         Bias= Scale*rand(1,N);
      %   fprintf('linearly scale the range of uniform distribution to %d\n',  Scale);
    else
         Weight=rand(Nfea,N)*2-1;
          Bias=rand(1,N);
    end
else if strcmp(RandomType,'Gaussian')
          Weight=randn(Nfea,N);
          Bias=randn(1,N);  
    else
        error('only Gaussian and Uniform are supported')
    end
end
Bias_train=repmat(Bias,Nsample,1);
H=trainX*Weight+Bias_train;

switch lower(ActivationFunction)
    case {'sig','sigmoid'}
      
        if Scale
          
            Saturating_threshold=[-4.6,4.6];
            Saturating_threshold_activate=[0,1];
            if Scalemode==1;
         
           [H,k,b]=Scale_feature(H,Saturating_threshold,Scale);
            elseif Scalemode==2
                
           [H,k,b]=Scale_feature_separately(H,Saturating_threshold,Scale);
            end
        end
        H = 1 ./ (1 + exp(-H));
    case {'sin','sine'}
       
         
        if Scale
   
           Saturating_threshold=[-pi/2,pi/2];
           Saturating_threshold_activate=[-1,1];
           if Scalemode==1
               
            [H,k,b]=Scale_feature(H,Saturating_threshold,Scale);
          elseif Scalemode==2
            [H,k,b]=Scale_feature_separately(H,Saturating_threshold,Scale);
           end
        end
        H = sin(H);    
    case {'hardlim'}
        
        H = double(hardlim(H));
    case {'tribas'}
       
         if Scale
          
            Saturating_threshold=[-1,1];
            Saturating_threshold_activate=[0,1];
            if Scalemode==1
         
            [H,k,b]=Scale_feature(H,Saturating_threshold,Scale);
            elseif Scalemode==2
         
             [H,k,b]=Scale_feature_separately(H,Saturating_threshold,Scale);
            end
        end
        H = tribas(H);
    case {'radbas'}
       
        if Scale
           
            Saturating_threshold=[-2.1,2.1];
            Saturating_threshold_activate=[0,1];
            if Scalemode==1
          
            [H,k,b]=Scale_feature(H,Saturating_threshold,Scale);
           elseif Scalemode==2
          
            [H,k,b]=Scale_feature_separately(H,Saturating_threshold,Scale);
            end
        end
        H = radbas(H);
           
    case {'sign'}
        H = sign(H);
end
if bias
   H=[H,ones(Nsample,1)]; 
end
if link
    
        switch Scalemode
            case 1 
            trainX_temp=trainX.*k+b;
            H=[H,trainX_temp];
            case 2
            [trainX_temp,ktr,btr]=Scale_feature_separately(trainX,Saturating_threshold_activate,Scale);
             H=[H,trainX_temp];
            otherwise
            H=[H,trainX];
        end
        
end
H(isnan(H))=0;
if mode==2
    
beta=pinv(H)*trainY_temp;
else if mode==1
        
    if ~isfield(option,'C')||isempty(C)
        C=0.1;
    end
    C=C;
    if N<Nsample
     beta=(eye(size(H,2))/C+H' * H) \ H'*trainY_temp;
    else
     beta=H'*((eye(size(H,1))/C+H* H') \ trainY_temp); 
    end
    else
      error('Unsupport mode, only Regularized least square and Moore-Penrose pseudoinverse are allowed. ')  
    end
end
trainY_temp=H*beta;
Y_temp=zeros(Nsample,1);


NumberofInputNeurons=size(trainX', 1);

weightt=[Weight,eye(NumberofInputNeurons,NumberofInputNeurons)];

ioWeight = sum(abs(weightt*beta),2);

csvwrite('ioweight',ioWeight)




% train_accuracy=length(find(Y_temp==trainY))/Nsample;

% test_accuracy=length(find(Yt_temp==testY))/numel(testY);


end

function [Output,k,b]=Scale_feature(Input,Saturating_threshold,ratio)
Min_value=min(min(Input));
Max_value=max(max(Input));
min_value=Saturating_threshold(1)*ratio;
max_value=Saturating_threshold(2)*ratio;
k=(max_value-min_value)/(Max_value-Min_value);
b=(min_value*Max_value-Min_value*max_value)/(Max_value-Min_value);
Output=Input.*k+b;
end

function [Output,k,b]=Scale_feature_separately(Input,Saturating_threshold,ratio)
nNeurons=size(Input,2);
k=zeros(1,nNeurons);
b=zeros(1,nNeurons);
Output=zeros(size(Input));
min_value=Saturating_threshold(1)*ratio;
max_value=Saturating_threshold(2)*ratio;
for i=1:nNeurons
Min_value=min(Input(:,i));
Max_value=max(Input(:,i));
k(i)=(max_value-min_value)/(Max_value-Min_value);
b(i)=(min_value*Max_value-Min_value*max_value)/(Max_value-Min_value);
Output(:,i)=Input(:,i).*k(i)+b(i);
end

end