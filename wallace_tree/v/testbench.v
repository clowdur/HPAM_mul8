// testbench for multipliers
//`default_nettype None

//`timescale 1ns/1ps

module testbench;

  //logic clk;
  logic [15:0] result; 
  logic [7:0] a, b;

  parameter T = 20;

  // define simulated clock if sequential logic is used
  // initial begin
  //   clk <= 0;
  //   forever #(T/2) clk <= ~clk;
  // end // initial clock

  wallaceTreeMultiplier8Bit dut (.*);

  initial begin
    $display("\n\n"); #20;
    a = 255; b = 255; #20;
    $display("inputs: a:%d, b:%d", a, b); #20;
    $display("result: %d\n", result); #20;
    a = 17; b = 17; #20;
    $display("inputs: a:%d, b:%d", a, b); #20;
    $display("result: %d\n", result); #20;
    a = 0; b = 0; #20;
    $display("inputs: a:%d, b:%d", a, b); #20;
    $display("result: %d\n", result); #20;
  end

endmodule