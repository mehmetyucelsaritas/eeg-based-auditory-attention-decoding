function [h, p] = createSubplotWithPlot(counter, totalRows, dataVector, plotTitle)
    % Automatically create the next subplot row
    h = subplot(totalRows, 1, counter);
    p = plot(dataVector);
    % Formatting
    title(plotTitle);
end
