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
inputoutweight=inputweight*bestweight;%ԭʼ�ľ������ 
csvwrite('inputoutweight',inputoutweight)
inoutweight=cumsum(inputoutweight,2);%����ÿ�����
csvwrite('inoutweight',inoutweight)
ioweight=inoutweight(:,end);%ȡ���һ��
% csvwrite('ioweight',ioweight)
sum1=sum(ioweight);%������Ԫ�صĺ�
ioweight=ioweight/sum1;%���й�һ������ 
% csvwrite('ioweight11',ioweight)
% for i=1:size(dx,1)
%  dx(i,1)=dx(i,1)/sum1;
% %  dx(i,1)=dx(i,1)*NumberofInputNeurons; %��ʦ�ø��ĵ�
% end
weightvalue=ioweight;%�����й���һ������ļ�Ȩֵ���ݸ����ڼ�Ȩ������ͳ����

