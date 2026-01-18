function updatePlots(startIdx, windowSize, dataCells, plotHandles, plotTitle)

    idx = startIdx:(startIdx+windowSize-1);
    
    for i = 1:length(plotHandles)
        p = plotHandles{i};
        dataVec = dataCells{i};
        
        % Update Y and X values
        p.YData = dataVec(idx);
        p.XData = idx;

        % Update axis limits
        xlim(p.Parent, [startIdx, startIdx + windowSize - 1]);

        % Only some plots need fixed y-limits
        if plotTitle{i} == "ICA-cleaned Data" || plotTitle{i} == "Asr-based artifact removal" % ICA and ASR
            ylim(p.Parent, [-100, 300]);
        end
    end
    drawnow;
end