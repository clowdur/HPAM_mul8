// Full Adder
// synthesizable
// Oliver Huang and Jared Yoder

module FA (input a, input b, input cin, output carry, output sum);

    // instantiating sum and carry primitives
    assign carry = (a & b) | (cin & ~a & ~b);
    assign sum = (a ^ b ^ cin);
endmodule