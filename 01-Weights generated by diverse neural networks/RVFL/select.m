load mmm.txt

weightvalue=mmm';
percent=0.9;


% weig=[
%    0.763    
%  0.6396   
%  0.5435   
%  0.4584   
%  0.382    
%  0.3148   
%  0.2498   
%  0.1899   
%  0.1372   
%  0.0935  
%  0.0587  
%  0.0272  
%  0       
% ];
% 
% ind=[
%     4
%   6
%  5
%  1
%   3
%  2
%  11
% 13
%   12
%   8
%  7
% 10
% 9
%     ];


[weig,ind]=sort(weightvalue,'descend');
weightva=[ind,weig];%按加权属性值递减排序后


csvwrite('afteraverageweight', weightva); 
num1=ceil(length(weightvalue)*percent)%向上取整，这是属性约简后属性的个数
csvwrite('attribute reduction  number', num1); 
weightvl=zeros(num1,2);
weightvl(1:num1,:)=weightva(1:num1,:);%降维后的加权值
% weightvl(num1+1:end,1)=weightva(num1+1:end,1);
csvwrite('attribute reduction matrix', weightvl); 
csvwrite('reduct index', sortrows(weightvl,1)'); 
