within ;
model test_sinus
  Modelica.Blocks.Sources.Sine sine(freqHz=1)
    annotation (Placement(transformation(extent={{-60,0},{-20,40}})));
  Modelica.Blocks.Interfaces.RealInput u_mul
    annotation (Placement(transformation(extent={{-142,-52},{-102,-12}})));
  Modelica.Blocks.Math.Product product1
    annotation (Placement(transformation(extent={{0,-20},{20,0}})));
  Modelica.Blocks.Interfaces.RealInput u_add
    annotation (Placement(transformation(extent={{-140,6},{-100,46}})));
  Modelica.Blocks.Math.Feedback feedback
    annotation (Placement(transformation(extent={{6,42},{26,62}})));
  Modelica.Blocks.Interfaces.RealOutput y_add
    annotation (Placement(transformation(extent={{94,42},{114,62}})));
  Modelica.Blocks.Interfaces.RealOutput y_mul
    annotation (Placement(transformation(extent={{98,-20},{118,0}})));
equation
  connect(sine.y, product1.u1) annotation (Line(points={{-18,20},{-8,20},{-8,-4},
          {-2,-4}}, color={0,0,127}));
  connect(u_mul, product1.u2) annotation (Line(points={{-122,-32},{-84,-32},{
          -84,-18},{-2,-18},{-2,-16}}, color={0,0,127}));
  connect(u_add, feedback.u1) annotation (Line(points={{-120,26},{-92,26},{-92,
          52},{8,52}}, color={0,0,127}));
  connect(sine.y, feedback.u2) annotation (Line(points={{-18,20},{-6,20},{-6,22},
          {16,22},{16,44}}, color={0,0,127}));
  connect(feedback.y, y_add)
    annotation (Line(points={{25,52},{104,52}}, color={0,0,127}));
  connect(product1.y, y_mul)
    annotation (Line(points={{21,-10},{108,-10}}, color={0,0,127}));
  annotation (uses(Modelica(version="3.2.3")));
end test_sinus;
