theta = 0:2:180;
cos_theta = cosd(theta);

zero_array = zeros(size(theta));

%----- modified softmax loss functions
%legend_strings = {};

fig = figure(1);
clf(fig);

hold on;

h = plot(theta, cos_theta, '-b','LineWidth', 2);
ax_hanlders = [h];
legend_strings = {'softmax(m=1)'};


%----- A-softmax loss functions
Fai = zeros(4,length(theta));

line_styles_0 = {'-r', '-g', '-m'};
style_cnt = 0;

for m=[2 3 4];
%     if m==3;
%         continue;
%     end    
    style_cnt = style_cnt + 1;
    for k=0:m-1;
        Fai(m,theta >= k * 180 / m & theta <= (k+1)*180/m) = (-1)^k * cosd(m*theta(theta >= k * 180 / m & theta <=(k+1)*180/m)) - 2 * k;
    end;
    
    legend_str = sprintf('large margin softmax(m=%d, \\lambda=0)', m);

    h = plot(theta, Fai(m,:), line_styles_0{style_cnt}, 'LineWidth',2);
    ax_hanlders(end+1) = h;
    legend_strings{end + 1, : } = legend_str;
end;

%m=4;
%lambda=5;
%plot(theta, (cos_theta * lambda + Fai(4, :)) / (1 + lambda),'LineWidth',2, 'Color', 'r');

line_styles_1 = {
    {'--r', '-+r', '-sr', '-or', '-dr'}; 
    {'--g', '-+g', '-sg', '-og', '-dg'};
    {'--m', '-+m', '-sm', '-om', '-dm'};};

style_cnt = 0;

for m=[2 3 4];
    style_cnt = style_cnt + 1;
%     if m<4;
%         continue;
%     end
    
    style_cnt2 = 0;
    
    for lambda=[1 2 5 10];
%    for lambda=[2 5 10];
        style_cnt2 = style_cnt2 + 1;

        legend_str = sprintf('Large Margin Softmax(m=%d, \\lambda=%d)', m, lambda);

        h = plot(theta, (cos_theta * lambda + Fai(m, :)) / (1 + lambda), line_styles_1{style_cnt}{style_cnt2}, 'LineWidth', 2);

        ax_hanlders(end+1) = h;
        legend_strings{end + 1, : } = legend_str;
    end  
end

%----- loss functions by zhaoyafei
% % 2*cos_theta - 1
% h = plot(theta, 2*cos_theta - 1, '--b','LineWidth',2);
% ax_hanlders(end+1) = h;
% legend_strings{end + 1, : } = 'Softmax^2/e';
% 
% % 3*cos_theta - 2
% h = plot(theta, 3*cos_theta - 2, '-+b','LineWidth',2);
% ax_hanlders(end+1) = h;
% legend_strings{end + 1, : } = 'Softmax^3/e^2';
% 
% % 4*cos_theta - 3
% h = plot(theta, 4*cos_theta - 3, '-sb','LineWidth',2);
% ax_hanlders(end+1) = h;
% legend_strings{end + 1, : } = 'Softmax^4/e^3';
% 
% % 5*cos_theta - 4
% h = plot(theta, 5*cos_theta - 4, '-ob','LineWidth',2);
% ax_hanlders(end+1) = h;
% legend_strings{end + 1, : } = 'Softmax^5/e^4';

line_styles_2 = {
    '--b', '-+b', '-sb', '-ob', '-db', ...
    '--k', '-+k', '-sk', '-ok', '-dk', ...
    '--y', '-+y', '-sy', '-oy', '-dy'
};

style_cnt = 0;
for alpha=[2 3 4];
%for alpha=[2 3 4 5];
%for alpha=[2 3 4 5 6 7 8];
%for alpha=[1.5 2 2.5 3 3.5 4];
    style_cnt = style_cnt + 1;

    legend_str = sprintf('Modified L-Softmax(%g*cos\\theta-%g)', alpha, alpha-1);

    h = plot(theta,  alpha*cos_theta - alpha+1, line_styles_2{style_cnt}, 'LineWidth', 2);

    ax_hanlders(end+1) = h;
    legend_strings{end + 1, : } = legend_str;
end  

hold off;

%legend('Softmax(m=1)','large margin Softmax(m=2, \lambda=0)','large margin Softmax(m=4, \lambda=0)', ['large margin Softmax(m=4, \lambda=' num2str(lambda) ')']);
legend(ax_hanlders, legend_strings, 'location', 'SouthWest');
grid on;
