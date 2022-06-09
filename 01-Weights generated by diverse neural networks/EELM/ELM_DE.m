function TrainingTime=ELM_DE(TrainingData_File, TestingData_File, Elm_Type, NumberofHiddenNeurons, ActivationFunction);
% minimization of a user-supplied function with respect to x(1:D),
% using the differential evolution (DE) algorithm of Rainer Storn
% (http://www.icsi.berkeley.edu/~storn/code.html)
% 
% Special thanks go to Ken Price (kprice@solano.community.net) and
% Arnold Neumaier (http://solon.cma.univie.ac.at/~neum/) for their
% valuable contributions to improve the code.
% 
% Strategies with exponential crossover, further input variable
% tests, and arbitrary function name implemented by Jim Van Zandt 
% <jrv@vanzandt.mv.com>, 12/97.
%
% Output arguments:
% ----------------
% bestmem        parameter vector with best solution
% bestval        best objective function value
% nfeval         number of function evaluations
%
% Iut arguments:  
% ---------------
%
% fname          string naming a function f(x,y) to minimize
% VTR            "Value To Reach". devec3 will stop its minimization
%                if either the maximum number of iterations "itermax"
%                is reached or the best parameter vector "bestmem" 
%                has found a value f(bestmem,y) <= VTR.
% D              number of parameters of the objective function 
% XVmin          vector of lower bounds XVmin(1) ... XVmin(D)
%                of initial population
%                *** note: these are not bound constraints!! ***
% XVmax          vector of upper bounds XVmax(1) ... XVmax(D)
%                of initial population
% y		        problem data vector (must remain fixed during the
%                minimization)
% NP             number of population members
% itermax        maximum number of iterations (generations)
% F              DE-stepsize F from interval [0, 2]
% CR             crossover probability constant from interval [0, 1]
% strategy       1 --> DE/best/1/exp           6 --> DE/best/1/bin
%                2 --> DE/rand/1/exp           7 --> DE/rand/1/bin
%                3 --> DE/rand-to-best/1/exp   8 --> DE/rand-to-best/1/bin
%                4 --> DE/best/2/exp           9 --> DE/best/2/bin
%                5 --> DE/rand/2/exp           else  DE/rand/2/bin
%                Experiments suggest that /bin likes to have a slightly
%                larger CR than /exp.
% refresh        intermediate output will be produced after "refresh"
%                iterations. No intermediate output will be produced
%                if refresh is < 1
%
%       The first four arguments are essential (though they have
%       default values, too). In particular, the algorithm seems to
%       work well only if [XVmin,XVmax] covers the region where the
%       global minimum is expected. DE is also somewhat sensitive to
%       the choice of the stepsize F. A good initial guess is to
%       choose F from interval [0.5, 1], e.g. 0.8. CR, the crossover
%       probability constant from interval [0, 1] helps to maintain
%       the diversity of the population and is rather uncritical. The
%       number of population members NP is also not very critical. A
%       good initial guess is 10*D. Depending on the difficulty of the
%       problem NP can be lower than 10*D or must be higher than 10*D
%       to achieve convergence.
%       If the parameters are correlated, high values of CR work better.
%       The reverse is true for no correlation.
%
% default values in case of missing input arguments:
% 	VTR = 1.e-6;
% 	D = 2; 
% 	XVmin = [-2 -2]; 
% 	XVmax = [2 2]; 
%	y=[];
% 	NP = 10*D; 
% 	itermax = 200; 
% 	F = 0.8; 
% 	CR = 0.5; 
% 	strategy = 7;
% 	refresh = 10; 
%
% Cost function:  	function result = f(x,y);
%                      	has to be defined by the user and is minimized
%			w.r. to  x(1:D).
%
% Example to find the minimum of the Rosenbrock saddle:
% ----------------------------------------------------
% Define f.m as:
%                    function result = f(x,y);
%                    result = 100*(x(2)-x(1)^2)^2+(1-x(1))^2;
%                    end
% Then type:
%
% 	VTR = 1.e-6;
% 	D = 2; 
% 	XVmin = [-2 -2]; 
% 	XVmax = [2 2]; 
% 	[bestmem,bestval,nfeval] = devec3("f",VTR,D,XVmin,XVmax);
%
% The same example with a more complete argument list is handled in 
% run1.m
%
% About devec3.m
% --------------
% Differential Evolution for MATLAB
% Copyright (C) 1996, 1997 R. Storn
% International Computer Science Institute (ICSI)
% 1947 Center Street, Suite 600
% Berkeley, CA 94704
% E-mail: storn@icsi.berkeley.edu
% WWW:    http://http.icsi.berkeley.edu/~storn
%
% devec is a vectorized variant of DE which, however, has a
% propertiy which differs from the original version of DE:
% 1) The random selection of vectors is performed by shuffling the
%    population array. Hence a certain vector can't be chosen twice
%    in the same term of the perturbation expression.
%
% Due to the vectorized expressions devec3 executes fairly fast
% in MATLAB's interpreter environment.
%
% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation; either version 1, or (at your option)
% any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details. A copy of the GNU 
% General Public License can be obtained from the 
% Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

%-----Check input variables---------------------------------------------
err=[];
XVmin=-1;
XVmax=1;
% if nargin<1, error('devec3 1st argument must be function name'); else 
%   if exist(fname)<1; err(1,length(err)+1)=1; end; end;
% if nargin<2, VTR = 1.e-6; else 
%   if length(VTR)~=1; err(1,length(err)+1)=2; end; end;
% if nargin<3, D = 2; else
%   if length(D)~=1; err(1,length(err)+1)=3; end; end; 
% if nargin<4, XVmin = [-2 -2];else
%   if length(XVmin)~=D; err(1,length(err)+1)=4; end; end; 
% if nargin<5, XVmax = [2 2]; else
%   if length(XVmax)~=D; err(1,length(err)+1)=5; end; end; 
% if nargin<6, y=[]; end; 
% if nargin<7, NP = 10*D; else
%   if length(NP)~=1; err(1,length(err)+1)=7; end; end; 
% if nargin<8, itermax = 200; else
%   if length(itermax)~=1; err(1,length(err)+1)=8; end; end; 
% if nargin<9, F = 0.8; else
%   if length(F)~=1; err(1,length(err)+1)=9; end; end;
% if nargin<10, CR = 0.5; else
%   if length(CR)~=1; err(1,length(err)+1)=10; end; end; 
% if nargin<11, strategy = 7; else
%   if length(strategy)~=1; err(1,length(err)+1)=11; end; end;
% if nargin<12, refresh = 10; else
%   if length(refresh)~=1; err(1,length(err)+1)=12; end; end; 
% if length(err)>0
%   fprintf(stdout,'error in parameter %d\n', err);
%   usage('devec3 (string,scalar,scalar,vector,vector,any,integer,integer,scalar,scalar,integer,integer)');    	
% end
REGRESSION=0;
CLASSIFIER=1;
Gain = 1;                                           %  Gain parameter for sigmoid

%%%%%%%%%%% Load training dataset
train_data=load(TrainingData_File);
T=train_data(:,1)';
P=train_data(:,2:size(train_data,2))';
clear train_data;                                   %   Release raw training data array

%%%%%%%%%%% Load testing dataset
test_data=load(TestingData_File);
TV.T=test_data(:,1)';
TV.P=test_data(:,2:size(test_data,2))';
clear test_data;                                    %   Release raw testing data array

NumberofTrainingData=size(P,2);
NumberofTestingData=size(TV.P,2);
NumberofInputNeurons=size(P,1);
NumberofValidationData = round(NumberofTestingData / 2);

if Elm_Type~=REGRESSION
    %%%%%%%%%%%% Preprocessing the data of classification
    sorted_target=sort(cat(2,T,TV.T),2);
    label=zeros(1,1);                               %   Find and save in 'label' class label from training and testing data sets
    label(1,1)=sorted_target(1,1);
    j=1;
    for i = 2:(NumberofTrainingData+NumberofTestingData)
        if sorted_target(1,i) ~= label(1,j)
            j=j+1;
            label(1,j) = sorted_target(1,i);
        end
    end
    number_class=j;
    NumberofOutputNeurons=number_class;
    
    %%%%%%%%%% Processing the targets of training
    temp_T=zeros(NumberofOutputNeurons, NumberofTrainingData);
    for i = 1:NumberofTrainingData
        for j = 1:number_class
            if label(1,j) == T(1,i)
                break; 
            end
        end
        temp_T(j,i)=1;
    end
    T=temp_T*2-1;

    %%%%%%%%%% Processing the targets of testing
    temp_TV_T=zeros(NumberofOutputNeurons, NumberofTestingData);
    for i = 1:NumberofTestingData
        for j = 1:number_class
            if label(1,j) == TV.T(1,i)
                break; 
            end
        end
        temp_TV_T(j,i)=1;
    end
    TV.T=temp_TV_T*2-1;
end                                                 %   end if of Elm_Type
clear temp_T;
clear temp_T;
VV.P = TV.P(:,1:NumberofValidationData);
VV.T = TV.T(:,1:NumberofValidationData);
TV.P(:,1:NumberofValidationData)=[];
TV.T(:,1:NumberofValidationData)=[];
NumberofTestingData = NumberofTestingData - NumberofValidationData;

%%%%%%%%%%% Calculate weights & biases
CR=0.8;
NP=200;
%NP=300;
D=NumberofHiddenNeurons*(NumberofInputNeurons+1);
itermax=20;
%itermax=30;
refresh=1;
strategy = 3;
F=1;
tolerance = 0.02;
start_time_validation=cputime;

if (NP < 5)
   NP=5;
   fprintf(1,' NP increased to minimal value 5\n');
end
if ((CR < 0) | (CR > 1))
   CR=0.5;
   fprintf(1,'CR should be from interval [0,1]; set to default value 0.5\n');
end
if (itermax <= 0)
   itermax = 200;
   %itermax = 300;
   fprintf(1,'itermax should be > 0; set to default value 200\n');
end
refresh = floor(refresh);

%-----Initialize population and some arrays-------------------------------

pop = zeros(NP,D); %initialize pop to gain speed

%----pop is a matrix of size NPxD. It will be initialized-------------
%----with random values between the min and max values of the---------
%----parameters-------------------------------------------------------

for i=1:NP
   pop(i,:) = XVmin + rand(1,D).*(XVmax - XVmin);
end

popold    = zeros(size(pop));     % toggle population
val       = zeros(1,NP);          % create and reset the "cost array"
bestmem   = zeros(1,D);           % best population member ever
bestmemit = zeros(1,D);           % best population member in iteration
nfeval    = 0;                    % number of function evaluations
brk    = 0;

%------Evaluate the best member after initialization----------------------

ibest   = 1;  % start with first population member
[val(1),OutputWeight]  = ELM_X(Elm_Type,pop(ibest,:),P,T,VV,NumberofHiddenNeurons);
bestval = val(1);                 % best objective function value so far
nfeval  = nfeval + 1;
bestweight = OutputWeight;
for i=2:NP                        % check the remaining members
  [val(i),OutputWeight] = ELM_X(Elm_Type,pop(i,:),P,T,VV,NumberofHiddenNeurons);
  nfeval  = nfeval + 1;
  if (val(i) < bestval)           % if member is better
     ibest   = i;                 % save its location
     bestval = val(i);
     bestweight = OutputWeight;
  end   
end
bestmemit = pop(ibest,:);         % best member of current iteration
bestvalit = bestval;              % best value of current iteration
bestmem = bestmemit;              % best member ever

%------DE-Minimization---------------------------------------------
%------popold is the population which has to compete. It is--------
%------static through one iteration. pop is the newly--------------
%------emerging population.----------------------------------------

pm1 = zeros(NP,D);              % initialize population matrix 1
pm2 = zeros(NP,D);              % initialize population matrix 2
pm3 = zeros(NP,D);              % initialize population matrix 3
pm4 = zeros(NP,D);              % initialize population matrix 4
pm5 = zeros(NP,D);              % initialize population matrix 5
bm  = zeros(NP,D);              % initialize bestmember  matrix
ui  = zeros(NP,D);              % intermediate population of perturbed vectors
mui = zeros(NP,D);              % mask for intermediate population
mpo = zeros(NP,D);              % mask for old population
rot = (0:1:NP-1);               % rotating index array (size NP)
rotd= (0:1:D-1);                % rotating index array (size D)
rt  = zeros(NP);                % another rotating index array
rtd = zeros(D);                 % rotating index array for exponential crossover
a1  = zeros(NP);                % index array
a2  = zeros(NP);                % index array
a3  = zeros(NP);                % index array
a4  = zeros(NP);                % index array
a5  = zeros(NP);                % index array
ind = zeros(4);

iter = 1;
while (~(iter > itermax) )
  popold = pop;                   % save the old population

  ind = randperm(4);              % index pointer array

  a1  = randperm(NP);             % shuffle locations of vectors
  rt = rem(rot+ind(1),NP);        % rotate indices by ind(1) positions
  a2  = a1(rt+1);                 % rotate vector locations
  rt = rem(rot+ind(2),NP);
  a3  = a2(rt+1);                
  rt = rem(rot+ind(3),NP);
  a4  = a3(rt+1);               
  rt = rem(rot+ind(4),NP);
  a5  = a4(rt+1);                

  pm1 = popold(a1,:);             % shuffled population 1
  pm2 = popold(a2,:);             % shuffled population 2
  pm3 = popold(a3,:);             % shuffled population 3
  pm4 = popold(a4,:);             % shuffled population 4
  pm5 = popold(a5,:);             % shuffled population 5

  for i=1:NP                      % population filled with the best member
    bm(i,:) = bestmemit;          % of the last iteration
  end

  mui = rand(NP,D) < CR;          % all random numbers < CR are 1, 0 otherwise

  if (strategy > 5)
    st = strategy-5;		  % binomial crossover
  else
    st = strategy;		  % exponential crossover
    mui=sort(mui');	          % transpose, collect 1's in each column
    for i=1:NP
      n=floor(rand*D);
      if n > 0
         rtd = rem(rotd+n,D);
         mui(:,i) = mui(rtd+1,i); %rotate column i by n
      end
    end
    mui = mui';			  % transpose back
  end
  mpo = mui < 0.5;                % inverse mask to mui

  if (st == 1)                      % DE/best/1
    ui = bm + F*(pm1 - pm2);        % differential variation
    ui = popold.*mpo + ui.*mui;     % crossover
  elseif (st == 2)                  % DE/rand/1
    ui = pm3 + F*(pm1 - pm2);       % differential variation
    ui = popold.*mpo + ui.*mui;     % crossover
  elseif (st == 3)                  % DE/rand-to-best/1
    ui = popold + F*(bm-popold) + F*(pm1 - pm2);        
    ui = popold.*mpo + ui.*mui;     % crossover
  elseif (st == 4)                  % DE/best/2
    ui = bm + F*(pm1 - pm2 + pm3 - pm4);  % differential variation
    ui = popold.*mpo + ui.*mui;           % crossover
  elseif (st == 5)                  % DE/rand/2
    ui = pm5 + F*(pm1 - pm2 + pm3 - pm4);  % differential variation
    ui = popold.*mpo + ui.*mui;            % crossover
  end

%-----Select which vectors are allowed to enter the new population------------
  for i=1:NP
    [tempval,OutputWeight] = ELM_X(Elm_Type,ui(i,:),P,T,VV,NumberofHiddenNeurons);   % check cost of competitor
    nfeval  = nfeval + 1;
    if (tempval <= val(i))  % if competitor is better than value in "cost array"
       pop(i,:) = ui(i,:);  % replace old vector with new one (for new iteration)
       val(i)   = tempval;  % save value in "cost array"

       %----we update bestval only in case of success to save time-----------
       if bestval-tempval>tolerance*bestval
           bestval = tempval;      % new best value
           bestmem = ui(i,:);      % new best parameter vector ever
           bestweight = OutputWeight;
       elseif abs(tempval-bestval)<tolerance*bestval    % if competitor better than the best one ever
           if norm(OutputWeight,2)<norm(bestweight,2)
                bestval = tempval;      % new best value
                bestmem = ui(i,:);      % new best parameter vector ever
                bestweight = OutputWeight;
           end
       end
    end
  end %---end for imember=1:NP

  bestmemit = bestmem;       % freeze the best member of this iteration for the coming 
                             % iteration. This is needed for some of the strategies.

%----Output section----------------------------------------------------------

  if (refresh > 0)
    if (rem(iter,refresh) == 0)
       fprintf(1,'Iteration: %d,  Best: %f,  F: %f,  CR: %f,  NP: %d\n',iter,bestval,F,CR,NP);
%        for n=1:D
%          fprintf(1,'best(%d) = %f\n',n,bestmem(n));
%        end
    end
  end
  iter = iter + 1;
end %---end while ((iter < itermax) ...
% csvwrite('NumberofHiddenNeurons',abs(NumberofHiddenNeurons))
% csvwrite('NumberofInputNeurons',abs(NumberofInputNeurons))
% csvwrite('NumberofOutputNeurons',abs(NumberofOutputNeurons))

Beta = mean(abs(OutputWeight))
NumberInputNeurons=size(P, 1);
NumberofTrainingData=size(P, 2);
NumberofTestingData=size(TV.P, 2);
Gain=1;
temp_weight_bias=reshape(bestmem, NumberofHiddenNeurons, NumberInputNeurons+1);
InputWeight=temp_weight_bias(:, 1:NumberInputNeurons);
csvwrite('bestweight',abs(bestweight))
csvwrite('Inputweight',abs(InputWeight'))
bestwe=bestweight;
inputwe=InputWeight';

%OutputWeight = weight(NumberofInputNeurons,NumberofHiddenNeurons,NumberofOutputNeurons,inputwe,bestwe); 
OutputWeight = weight(NumberofInputNeurons,NumberofHiddenNeurons,NumberofOutputNeurons); 
csvwrite('outputweight',OutputWeight)

end_time_validation=cputime;
TrainingTime=end_time_validation-start_time_validation;

%%%%%%%%%%%%% Testing the performance of the best population
% Beta = mean(abs(OutputWeight))
% NumberInputNeurons=size(P, 1);
% NumberofTrainingData=size(P, 2);
% NumberofTestingData=size(TV.P, 2);
% Gain=1;
% temp_weight_bias=reshape(bestmem, NumberofHiddenNeurons, NumberInputNeurons+1);
% InputWeight=temp_weight_bias(:, 1:NumberInputNeurons);
% BiasofHiddenNeurons=temp_weight_bias(:,NumberInputNeurons+1);
% tempH=InputWeight*P;
% ind=ones(1,NumberofTrainingData);
% BiasMatrix=BiasofHiddenNeurons(:,ind);      %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
% tempH=tempH+BiasMatrix;
% clear BiasMatrix
% H = 1 ./ (1 + exp(-Gain*tempH));
% clear tempH;
% % OutputWeight=pinv(H') * T';
% Y=(H' * bestweight)';
% tempH_test=InputWeight*TV.P;
% ind=ones(1,NumberofTestingData);
% BiasMatrix=BiasofHiddenNeurons(:,ind);      %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
% tempH_test=tempH_test + BiasMatrix;
% H_test = 1 ./ (1 + exp(-Gain*tempH_test));
% TY=(H_test' * bestweight)';
% NumberofHiddenNeurons;
% NumberofOutputNeurons;
% NumberofInputNeurons;
% if Elm_Type == 0
%     TrainingAccuracy=sqrt(mse(T - Y))
%     TestingAccuracy=sqrt(mse(TV.T - TY));            %   Calculate testing accuracy (RMSE) for regression case
% end

% if Elm_Type == 1
% %%%%%%%%%% Calculate training & testing classification accuracy
%     MissClassificationRate_Testing=0;
%     MissClassificationRate_Training=0;
%     for i = 1 : size(T, 2)
%         [x, label_index_expected]=max(T(:,i));
%         [x, label_index_actual]=max(Y(:,i));
%         if label_index_actual~=label_index_expected
%             MissClassificationRate_Training=MissClassificationRate_Training+1;
%         end
%     end
%     TrainingAccuracy=1-MissClassificationRate_Training/size(T,2)
%     for i = 1 : size(TV.T, 2)
%         [x, label_index_expected]=max(TV.T(:,i));
%         [x, label_index_actual]=max(TY(:,i));
%         if label_index_actual~=label_index_expected
%             MissClassificationRate_Testing=MissClassificationRate_Testing+1;
%         end
%     end
%     TestingAccuracy=1-MissClassificationRate_Testing/size(TV.T,2);
% end
% csvwrite('bestweight',abs(bestweight))
% csvwrite('Inputweight',abs(InputWeight'))
% bestwe=bestweight;
% inputwe=InputWeight';
% 
% OutputWeight = weight(NumberofInputNeurons,NumberofHiddenNeurons,NumberofOutputNeurons,inputwe,bestwe); 
% csvwrite('outputweight',OutputWeight)

% train_data1=load(TrainingData_File);
% test_data1=load(TestingData_File);
% %OutputWeight=zeros(NumberofInputNeurons,1) ;
% %for i=1:size(OutputWeight,1)
% %  OutputWeight(i,1)=1;
% %end
% distanceof =distance(train_data1,test_data1,OutputWeight);
%distanceof =distance(train_data1,test_data1);
% csvwrite('result', distanceof); 
% correct=0;
% incorrect=0;
% for i=1:size(test_data1,1)
%     if test_data1(i,1)==distanceof(i,2);
%         correct=correct+1;
%     else 
%        incorrect=incorrect+1;     
%     end   
% end    
%  disp('训练正确的个数为：')
%  correct
%  disp('不匹配的个数为：')
%  incorrect
%  disp('正确率：')
%  TestingAccuracy= correct/(correct+incorrect)
% clear all;
% global xyz
% xyz=correct/(correct+incorrect);

%csvwrite('jieguo', aa); 
%装载特征权重文件

% xlswrite('B.xls',aa);
%C=[aa g];
% a=load('jieguo');
 % xlswrite('A.xls',a);

