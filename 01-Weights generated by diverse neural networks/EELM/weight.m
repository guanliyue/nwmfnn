function weightvalue =weight(NumberofInputNeurons,NumberofHiddenNeurons,NumberofOutputNeurons)
%sum1=0;
dx=zeros(NumberofInputNeurons,1);
load inputweight;
load bestweight;

% for j=1:NumberofInputNeurons
%         d=0;
%     for m=1:NumberofHiddenNeurons
%         for n=1:NumberofOutputNeurons
%           d=0;
%           %d=abs(inputwe(j,m)*bestwe(m,n))+d;
%           d=abs(inputwe(j,m)*bestwe(m,n))+d;
%         end
%     end
%     dx(j,:)=d;
%     sum1=d+sum1;
% end
inputoutweight=zeros(NumberofInputNeurons,NumberofOutputNeurons);
inoutweight=zeros(NumberofInputNeurons,NumberofOutputNeurons);
ioweight=zeros(NumberofInputNeurons,1);
inputoutweight=inputweight*bestweight;%原始的矩阵想成 
csvwrite('inputoutweight',inputoutweight)
inoutweight=cumsum(inputoutweight,2);%按列每列相加
csvwrite('inoutweight',inoutweight)
ioweight=inoutweight(:,end);%取最后一列
% csvwrite('ioweight',ioweight)
sum1=sum(ioweight);%求所有元素的和
ioweight=ioweight/sum1;%进行归一化处理 
% csvwrite('ioweight11',ioweight)
% for i=1:size(dx,1)
%  dx(i,1)=dx(i,1)/sum1;
% %  dx(i,1)=dx(i,1)*NumberofInputNeurons; %老师让更改的
% end
weightvalue=ioweight;%将进行过归一化处理的加权值传递给近邻加权的数组统计中

