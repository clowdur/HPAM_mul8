// Half Adder
// synthesizable
// Oliver Huang and Jared Yoder

module HA (input a, input b, output carry, output sum);

    // instantiating sum and carry primitives
    assign carry = a & b;
    assign sum = a ^ b;
endmodule