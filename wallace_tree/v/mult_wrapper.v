// adding flip flops on input and output of multiplier for timing analysis

module mult_wrapper (output logic [15:0] Y, input logic [7:0] inA, input logic [7:0] inB, input logic clk);

  logic [15:0] result;
  logic [7:0] a, b;

  always_ff @(posedge clk) begin
    a <= inA;
    b <= inB;
    Y <= result;
  end

  wallaceTreeMultiplier8Bit multiply (.result(result), .a(a), .b(b));

endmodule