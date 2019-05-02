function [correlation] = output_plot(outcome,treatment,weight,coef,state_list)
[T,N] = size(outcome);
correlation = zeros(T,2,N);
for i=1:N
    [y1,y2] = gsc_fit(outcome,treatment,weight,coef,i,N);
    y1 = hpfilter(y1,14400);
    y2 = hpfilter(y2,14400);
    correlation(:,:,i) = [y1,y2];
    %{
    f = figure('visible','off');
    plot(y1)
    hold on
    plot(y2)
    ylabel('Mobility Rate');
    grid on
    grid minor
    legend(state_list(i),'GSC Control');
    title(state_list(i));
    saveas(f,strcat(num2str(i),'_',state_list(i),'.png'));
    %}
end




end