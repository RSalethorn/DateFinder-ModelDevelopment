class DateArrayPlacement:
    def __init__(self):
        # Positions of parts of date 
        #(Check google sheet for information)
        self.alt_clock_h_space_b = 0
        self.alt_clock_h = 1
        self.alt_clock_h_space_a = 2

        self.alt_clock_sep_1_space_b = 3
        self.alt_clock_sep_1 = 4
        self.alt_clock_sep_1_space_a = 5

        self.alt_clock_m_space_b = 6
        self.alt_clock_m = 7
        self.alt_clock_m_space_a = 8

        self.alt_clock_sep_2_space_b = 9
        self.alt_clock_sep_2 = 10
        self.alt_clock_sep_2_space_a = 11

        self.alt_clock_s_space_b = 12
        self.alt_clock_s = 13
        self.alt_clock_s_space_a = 14

        self.alt_clock_sep_3_space_b = 15
        self.alt_clock_sep_3 = 16
        self.alt_clock_sep_3_space_a = 17

        self.alt_clock_ms_space_b = 18
        self.alt_clock_ms = 19
        self.alt_clock_ms_space_a = 20

        self.alt_ampm_space_b = 21
        self.alt_ampm = 22
        self.alt_ampm_space_a = 23

        self.alt_tz_val_space_b = 24
        self.alt_tz_val = 25
        self.alt_tz_val_space_a = 26

        self.alt_tz_label_space_b = 27
        self.alt_tz_label = 28
        self.alt_tz_label_space_a = 29

        self.dmy_1_space_b = 30
        self.dmy_1 = 31
        self.dmy_1_space_a = 32

        self.dmy_1_ordinal_space_b = 33
        self.dmy_1_ordinal = 34
        self.dmy_1_ordinal_space_a = 35

        self.seperator_1_space_b = 36
        self.seperator_1 = 37
        self.seperator_1_space_a = 38

        self.dmy_2_space_b = 39
        self.dmy_2 = 40
        self.dmy_2_space_a = 41

        self.dmy_2_ordinal_space_b = 42
        self.dmy_2_ordinal = 43
        self.dmy_2_ordinal_space_a = 44

        self.seperator_2_space_b = 45
        self.seperator_2 = 46
        self.seperator_2_space_a = 47

        self.dmy_3_space_b = 48
        self.dmy_3 = 49
        self.dmy_3_space_a = 50

        self.dmy_3_ordinal_space_b = 51
        self.dmy_3_ordinal = 52
        self.dmy_3_ordinal_space_a = 53

        self.clock_h_space_b = 54
        self.clock_h = 55
        self.clock_h_space_a = 56

        self.clock_sep_1_space_b = 57
        self.clock_sep_1 = 58
        self.clock_sep_1_space_a = 59

        self.clock_m_space_b = 60
        self.clock_m = 61
        self.clock_m_space_a = 62

        self.clock_sep_2_space_b = 63
        self.clock_sep_2 = 64
        self.clock_sep_2_space_a = 65

        self.clock_s_space_b = 66
        self.clock_s = 67
        self.clock_s_space_a = 68

        self.clock_sep_3_space_b = 69
        self.clock_sep_3 = 70
        self.clock_sep_3_space_a = 71

        self.clock_ms_space_b = 72
        self.clock_ms = 73
        self.clock_ms_space_a = 74

        self.ampm_space_b = 75
        self.ampm = 76
        self.ampm_space_a = 77

        self.tz_val_space_b = 78
        self.tz_val = 79
        self.tz_val_space_a = 80

        self.tz_label_space_b = 81
        self.tz_label = 82
        self.tz_label_space_a = 83

        # Default clock parts
        self.default_clock_parts = [
            self.clock_h_space_b, self.clock_h, self.clock_h_space_a,
            self.clock_sep_1_space_b, self.clock_sep_1, self.clock_sep_1_space_a,
            self.clock_m_space_b, self.clock_m, self.clock_m_space_a,
            self.clock_sep_2_space_b, self.clock_sep_2, self.clock_sep_2_space_a,
            self.clock_s_space_b, self.clock_s, self.clock_s_space_a,
            self.clock_sep_3_space_b, self.clock_sep_3, self.clock_sep_3_space_a,
            self.clock_ms_space_b, self.clock_ms, self.clock_ms_space_a,
            self.ampm_space_b, self.ampm, self.ampm_space_a,
            self.tz_val_space_b, self.tz_val, self.tz_val_space_a,
            self.tz_label_space_b, self.tz_label, self.tz_label_space_a
        ]

        # Alt clock parts
        self.alt_clock_parts = [
            self.alt_clock_h_space_b, self.alt_clock_h, self.alt_clock_h_space_a,
            self.alt_clock_sep_1_space_b, self.alt_clock_sep_1, self.alt_clock_sep_1_space_a,
            self.alt_clock_m_space_b, self.alt_clock_m, self.alt_clock_m_space_a,
            self.alt_clock_sep_2_space_b, self.alt_clock_sep_2, self.alt_clock_sep_2_space_a,
            self.alt_clock_s_space_b, self.alt_clock_s, self.alt_clock_s_space_a,
            self.alt_clock_sep_3_space_b, self.alt_clock_sep_3, self.alt_clock_sep_3_space_a,
            self.alt_clock_ms_space_b, self.alt_clock_ms, self.alt_clock_ms_space_a,
            self.alt_ampm_space_b, self.alt_ampm, self.alt_ampm_space_a,
            self.alt_tz_val_space_b, self.alt_tz_val, self.alt_tz_val_space_a,
            self.alt_tz_label_space_b, self.alt_tz_label, self.alt_tz_label_space_a
        ]