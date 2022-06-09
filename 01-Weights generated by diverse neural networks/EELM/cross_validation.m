function AverageTrainingTime = cross_validation
start_time_train=cputime;
    load datasets.dt 
    dataset=datasets;
    weight=[];
%     LearnTime=[];
    [M,N]=size(dataset);
 
    indices=crossvalind('Kfold',M,10)

    
    
    for k=1:10 
        test12 = (indices == k); 
        train = ~test12; 
        train_data=dataset(train,:);

        test_data=dataset(test12,:);

 
  
    for i=1:size(  dataset,2)
        if max(  dataset(:,i))~=min( dataset(:,i))
              dataset(:,i)=( dataset(:,i)-min( dataset(:,i)))/(max( dataset(:,i))-min( dataset(:,i)));
        else
              dataset(:,i)=0;
        end
    end   
    
    P1=train_data(:,1:size(train_data,2)-1);
    T1=train_data(:,size(train_data,2));
    
    X=test_data(:,1:size(test_data,2)-1);
    Y=test_data(:,size(test_data,2));
    
      fid = fopen('datasets_train','w');
        for i=1:size(P1,1)
            fprintf(fid,'%2.8f ',T1(i,1));
            for j=1:size(P1,2)

                fprintf(fid,' %2.8f', P1(i,j));   
            end
            fprintf(fid,'\n');
        end
      fclose(fid);

      fid = fopen('datasets_test','w');   
      for i=1:size(X,1)
         fprintf(fid,'%2.8f ',Y(i,1));
            for j=1:size(X,2)
              fprintf(fid,' %2.8f', X(i,j));   
            end
              fprintf(fid,'\n');
      end
      fclose(fid);
      
      ELM_DE('datasets_train','datasets_test',1,100);
   
    load outputweight;
    weight=horzcat( weight, outputweight);

    
    end
    end_time_train=cputime;
TrainingTime=end_time_train-start_time_train

    csvwrite('tenweight',weight);
    length(weight)
lll=mean(weight,2);
lll=lll/sum(lll);
csvwrite('weights.txt', lll')


  