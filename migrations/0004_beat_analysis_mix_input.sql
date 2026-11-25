ALTER TABLE beat_analyses DROP CONSTRAINT IF EXISTS beat_analyses_input_stem_check;
ALTER TABLE beat_analyses ADD CONSTRAINT beat_analyses_input_stem_check
    CHECK (input_stem IN ('drums', 'mix'));
