function AverageTrainingTime = cross_validation
start_time_train=cputime;

    load datasets.dt 
    dataset=datasets;
    tenweight=[];
    [M,N]=size(dataset);
 
    indices=crossvalind('Kfold',M,10)

    
    
    for k=1:10 
        test12 = (indices == k);
        train = ~test12;
        train_data=dataset(train,:);
  
    for i=1:size(  dataset,2)
        if max(  dataset(:,i))~=min( dataset(:,i))
              dataset(:,i)=( dataset(:,i)-min( dataset(:,i)))/(max( dataset(:,i))-min( dataset(:,i)));
        else
              dataset(:,i)=0;
        end
    end   
    
    P1=train_data(:,1:size(train_data,2)-1);
    T1=train_data(:,size(train_data,2));

    
      fid = fopen('datasets_train','w');
        for i=1:size(P1,1)
            fprintf(fid,'%2.8f ',T1(i,1));
            for j=1:size(P1,2)
%            fprintf(fid,' %d:%2.8f',j, P1(i,j));    %   for SVM
                fprintf(fid,' %2.8f', P1(i,j));    %   for ELM
            end
            fprintf(fid,'\n');
        end
      fclose(fid);

    ELM('datasets_train',100,'sig');   
    load finalweight;
    tenweight=horzcat(tenweight, finalweight);
    

    
    end
        end_time_train=cputime;
TrainingTime=end_time_train-start_time_train
csvwrite('TrainingTime.txt',TrainingTime);
    csvwrite('tenweight',tenweight); 
    
    length(tenweight)
lll=mean(tenweight,2);
lll=lll/sum(lll);

csvwrite('weights.txt', lll')



  