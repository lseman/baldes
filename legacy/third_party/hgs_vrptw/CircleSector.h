#ifndef CIRCLESECTOR_H
#define CIRCLESECTOR_H

// Data structure to represent circle sectors
// Angles are measured in [0,65535] instead of [0,359], for faster modulo operations (2^16 = 65536).
// Credit to Fabian Giesen at
// "https://web.archive.org/web/20200912191950/https://fgiesen.wordpress.com/2015/09/24/intervals-in-modular-arithmetic/"
// for implementation tips regarding interval overlaps in modular arithmetic.
struct CircleSector {
    int start; // The angle where the circle sector starts
    int end;   // The angle where the circle sector ends

    // Calculate the positive modulo 65536 of `i` using a bitwise AND operation.
    static inline int positive_mod(int i) {
        return i & 0xFFFF; // Equivalent to i % 65536, but faster.
    }

    // Initialize a circle sector from a single point
    // The function `extend(int point)` can be used to grow the circle sector.
    void initialize(int point) {
        start = point;
        end   = point;
    }

    // Test if a point is enclosed within the circle sector
    bool isEnclosed(int point) {
        int start_diff = positive_mod(point - start);
        int end_diff   = positive_mod(end - start);
        return start_diff <= end_diff;
    }

    // Test overlap between two circle sectors with tolerance
    static bool overlap(const CircleSector &sector1, const CircleSector &sector2, const int tolerance) {
        int sector1_size = positive_mod(sector1.end - sector1.start) + tolerance;
        int sector2_size = positive_mod(sector2.end - sector2.start) + tolerance;
        return ((positive_mod(sector2.start - sector1.start) <= sector1_size) ||
                (positive_mod(sector1.start - sector2.start) <= sector2_size));
    }

    // Extend the circle sector to include an additional point
    // Greedily extend the sector to be the smallest possible sector that includes the point.
    void extend(int point) {
        if (!isEnclosed(point)) {
            // Greedily decide whether to extend the start or the end
            int extend_end_diff   = positive_mod(point - end);
            int extend_start_diff = positive_mod(start - point);

            if (extend_end_diff <= extend_start_diff) {
                end = point;
            } else {
                start = point;
            }
        }
    }
};

#endif
